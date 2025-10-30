"""Neo4j Aura backend for Drop in a Pond ego graphs.

This module provides an alternative storage backend using Neo4j graph database,
while maintaining compatibility with the existing EgoData structure used by
the analysis pipeline.

Graph Schema:
- Person nodes: {id, name, is_focal, graph_name, embedding: LIST<FLOAT>}
- Phrase nodes: {text, weight, last_updated, embedding: LIST<FLOAT>}
- Relationships:
  - (Person)-[:HAS_PHRASE]->(Phrase)
  - (Person)-[:CONNECTED_TO {actual, channels}]->(Person)
  - (Person)-[:HAS_CAPABILITY {capability}]->(Person)
  - (Person)-[:HAS_NOTE {date, content}]->(Person)
  - (Person)-[:AVAILABILITY {date, score, content}]->(Person)

Contact points are stored as Event nodes:
- Event nodes: {type, date, content, graph_name}
- (Event)-[:INVOLVES]->(Person)

Embedding Options:
1. OpenAI (recommended): Uses Neo4j genai.vector.encode() with text-embedding-3-small
   - Dimensions: 1536 (or configurable)
   - Quality: MTEB 62.3 (significantly better)
   - Cost: ~$0.0004 per typical ego graph

2. Sentence-Transformers: Local computation with all-MiniLM-L6-v2
   - Dimensions: 384
   - Quality: MTEB 56.3 (good, 10% lower)
   - Cost: Free (compute only)

Vector Indexes created automatically for similarity search.
"""

import os
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
from neo4j import GraphDatabase, Driver
from datetime import datetime

from .storage import EgoData


class Neo4jConnection:
    """Manages Neo4j Aura connection with environment-based configuration."""

    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None,
                 password: Optional[str] = None):
        """Initialize connection to Neo4j Aura.

        Args:
            uri: Neo4j URI (e.g., "neo4j+s://xxx.databases.neo4j.io")
                 Falls back to NEO4J_URI environment variable
                 Can also auto-construct from NEO4J_ID if NEO4J_URI not set
            username: Neo4j username (falls back to NEO4J_USERNAME env var)
            password: Neo4j password (falls back to NEO4J_PASSWORD env var)
        """
        # Try to get URI, or construct from NEO4J_ID if available
        self.uri = uri or os.getenv('NEO4J_URI')
        if not self.uri:
            neo4j_id = os.getenv('NEO4J_ID')
            if neo4j_id:
                self.uri = f"neo4j+s://{neo4j_id}.databases.neo4j.io"

        self.username = username or os.getenv('NEO4J_USERNAME')
        self.password = password or os.getenv('NEO4J_PASSWORD')

        if not all([self.uri, self.username, self.password]):
            raise ValueError(
                "Neo4j credentials not provided. Set NEO4J_ID (or NEO4J_URI), "
                "NEO4J_USERNAME, and NEO4J_PASSWORD environment variables or pass them explicitly."
            )

        self.driver: Optional[Driver] = None

    def __enter__(self) -> Driver:
        """Context manager entry - establish connection."""
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )
        self.driver.verify_connectivity()
        return self.driver

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        if self.driver:
            self.driver.close()


def _create_vector_indexes(session, graph_name: str, embedding_dim: int):
    """Create vector indexes for Person and Phrase embeddings if they don't exist.

    Args:
        session: Neo4j session
        graph_name: Name of the graph (used in index name)
        embedding_dim: Dimension of embedding vectors (384 or 1536)
    """
    # Index for Person node embeddings
    session.run(f"""
        CREATE VECTOR INDEX person_embedding_{graph_name} IF NOT EXISTS
        FOR (p:Person)
        ON p.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
    """)

    # Index for Phrase node embeddings
    session.run(f"""
        CREATE VECTOR INDEX phrase_embedding_{graph_name} IF NOT EXISTS
        FOR (phrase:Phrase)
        ON phrase.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
    """)


def load_ego_graph_from_neo4j(
    graph_name: str,
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> EgoData:
    """Load an ego graph from Neo4j Aura and return EgoData structure.

    Embeddings are loaded directly from Neo4j (no external embedding service needed).

    Args:
        graph_name: Name of the ego graph (stored as graph_name property on nodes)
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password

    Returns:
        EgoData structure ready for semantic flow analysis
    """
    with Neo4jConnection(uri, username, password) as driver:
        with driver.session() as session:
            # Find focal node
            result = session.run("""
                MATCH (focal:Person {graph_name: $graph_name, is_focal: true})
                RETURN focal.id as id, focal.name as name
            """, graph_name=graph_name)

            focal_record = result.single()
            if not focal_record:
                raise ValueError(f"No focal node found for graph '{graph_name}'")

            focal_id = focal_record['id']

            # Get all nodes with embeddings
            result = session.run("""
                MATCH (p:Person {graph_name: $graph_name})
                RETURN p.id as id, p.name as name, p.embedding as embedding
                ORDER BY p.id
            """, graph_name=graph_name)

            nodes = []
            names = {}
            embeddings = {}

            for record in result:
                node_id = record['id']
                nodes.append(node_id)
                names[node_id] = record['name']

                # Convert Neo4j list to numpy array
                if record['embedding']:
                    embeddings[node_id] = np.array(record['embedding'], dtype=np.float32)
                else:
                    # If no embedding, use zero vector (will be computed later)
                    embeddings[node_id] = None

            # Get edges
            result = session.run("""
                MATCH (p1:Person {graph_name: $graph_name})
                      -[r:CONNECTED_TO]->(p2:Person {graph_name: $graph_name})
                RETURN p1.id as source, p2.id as target,
                       r.actual as actual, r.channels as channels
            """, graph_name=graph_name)

            edges = []
            for record in result:
                edge_data = {'actual': record['actual'] or 0.0}
                if record['channels']:
                    edge_data['channels'] = record['channels']
                edges.append((record['source'], record['target'], edge_data))

            return EgoData(
                nodes=nodes,
                focal=focal_id,
                embeddings=embeddings,
                edges=edges,
                names=names
            )


def save_ego_graph_to_neo4j(
    graph_name: str,
    ego_data: EgoData,
    node_details: Dict[str, Dict],
    contact_points: Optional[Dict] = None,
    uri: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    openai_token: Optional[str] = None,
    embedding_model: Literal['openai', 'local'] = 'openai',
    openai_model: str = 'text-embedding-3-small',
    openai_dimensions: Optional[int] = None,
    clear_existing: bool = True
) -> None:
    """Save an ego graph to Neo4j Aura with embeddings.

    Args:
        graph_name: Name identifier for the graph
        ego_data: EgoData structure with nodes, edges, embeddings
        node_details: Dict mapping node_id to details (phrases, capabilities, notes, etc.)
        contact_points: Optional dict with past/present/potential contact events
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        openai_token: OpenAI API token (required if embedding_model='openai')
        embedding_model: 'openai' (recommended) or 'local' (sentence-transformers)
        openai_model: OpenAI model name (default: text-embedding-3-small)
        openai_dimensions: Optional dimension reduction for OpenAI embeddings
        clear_existing: If True, delete existing graph with same name first
    """
    if embedding_model == 'openai':
        if not openai_token:
            openai_token = os.getenv('OPENAI_API_KEY')
        if not openai_token:
            raise ValueError(
                "OpenAI token required for embedding_model='openai'. "
                "Set OPENAI_API_KEY environment variable or pass openai_token parameter."
            )

    # Determine embedding dimension
    if embedding_model == 'openai':
        if openai_dimensions:
            embedding_dim = openai_dimensions
        elif openai_model == 'text-embedding-3-small':
            embedding_dim = 1536
        elif openai_model == 'text-embedding-3-large':
            embedding_dim = 3072
        else:
            embedding_dim = 1536  # Default
    else:
        # Local sentence-transformers
        embedding_dim = 384

    with Neo4jConnection(uri, username, password) as driver:
        with driver.session() as session:
            # Clear existing graph if requested
            if clear_existing:
                session.run("""
                    MATCH (n {graph_name: $graph_name})
                    DETACH DELETE n
                """, graph_name=graph_name)

            # Create Person nodes
            for node_id in ego_data.nodes:
                is_focal = (node_id == ego_data.focal)
                name = ego_data.names.get(node_id, node_id)

                session.run("""
                    CREATE (p:Person {
                        graph_name: $graph_name,
                        id: $id,
                        name: $name,
                        is_focal: $is_focal
                    })
                """, graph_name=graph_name, id=node_id, name=name, is_focal=is_focal)

            # Add phrases and compute embeddings
            for node_id, details in node_details.items():
                phrases = details.get('phrases', [])

                if embedding_model == 'openai':
                    # Batch compute phrase embeddings using Neo4j genai procedures
                    phrase_texts = [p['text'] for p in phrases]

                    if phrase_texts:
                        # Encode all phrases in a batch
                        config = {'token': openai_token, 'model': openai_model}
                        if openai_dimensions:
                            config['dimensions'] = openai_dimensions

                        result = session.run("""
                            CALL genai.vector.encodeBatch($phrases, 'OpenAI', $config)
                            YIELD index, vector
                            RETURN index, vector
                        """, phrases=phrase_texts, config=config)

                        phrase_embeddings = {r['index']: r['vector'] for r in result}

                        # Create phrase nodes with embeddings
                        for i, phrase in enumerate(phrases):
                            session.run("""
                                MATCH (p:Person {graph_name: $graph_name, id: $node_id})
                                CREATE (phrase:Phrase {
                                    graph_name: $graph_name,
                                    text: $text,
                                    weight: $weight,
                                    last_updated: $last_updated
                                })
                                CALL db.create.setNodeVectorProperty(phrase, 'embedding', $embedding)
                                CREATE (p)-[:HAS_PHRASE]->(phrase)
                            """,
                            graph_name=graph_name,
                            node_id=node_id,
                            text=phrase['text'],
                            weight=phrase.get('weight', 1.0),
                            last_updated=phrase.get('last_updated', datetime.now().isoformat()),
                            embedding=phrase_embeddings[i]
                            )

                    # Compute Person embedding as weighted mean of phrase embeddings
                    if phrases:
                        session.run("""
                            MATCH (p:Person {graph_name: $graph_name, id: $node_id})
                                  -[:HAS_PHRASE]->(phrase:Phrase)
                            WITH p,
                                 sum(phrase.weight) as total_weight,
                                 [i IN range(0, size(phrase.embedding)-1) |
                                     sum([inner_phrase IN collect(phrase) |
                                         inner_phrase.embedding[i] * inner_phrase.weight]) /
                                     sum([inner_phrase IN collect(phrase) | inner_phrase.weight])
                                 ] as mean_embedding
                            CALL db.create.setNodeVectorProperty(p, 'embedding', mean_embedding)
                        """, graph_name=graph_name, node_id=node_id)

                else:
                    # Local embeddings - use pre-computed from ego_data
                    for phrase in phrases:
                        session.run("""
                            MATCH (p:Person {graph_name: $graph_name, id: $node_id})
                            CREATE (phrase:Phrase {
                                graph_name: $graph_name,
                                text: $text,
                                weight: $weight,
                                last_updated: $last_updated
                            })
                            CREATE (p)-[:HAS_PHRASE]->(phrase)
                        """,
                        graph_name=graph_name,
                        node_id=node_id,
                        text=phrase['text'],
                        weight=phrase.get('weight', 1.0),
                        last_updated=phrase.get('last_updated', datetime.now().isoformat())
                        )

                    # Store pre-computed Person embedding
                    if node_id in ego_data.embeddings and ego_data.embeddings[node_id] is not None:
                        embedding_list = ego_data.embeddings[node_id].tolist()
                        session.run("""
                            MATCH (p:Person {graph_name: $graph_name, id: $node_id})
                            CALL db.create.setNodeVectorProperty(p, 'embedding', $embedding)
                        """, graph_name=graph_name, node_id=node_id, embedding=embedding_list)

                # Add capabilities
                capabilities = details.get('capabilities', [])
                for capability in capabilities:
                    session.run("""
                        MATCH (p:Person {graph_name: $graph_name, id: $node_id})
                        MERGE (p)-[:HAS_CAPABILITY {capability: $capability}]->(p)
                    """, graph_name=graph_name, node_id=node_id, capability=capability)

                # Add notes
                notes = details.get('notes', [])
                for note in notes:
                    session.run("""
                        MATCH (p:Person {graph_name: $graph_name, id: $node_id})
                        CREATE (p)-[:HAS_NOTE {
                            date: $date,
                            content: $content
                        }]->(p)
                    """,
                    graph_name=graph_name,
                    node_id=node_id,
                    date=note['date'],
                    content=note['content']
                    )

                # Add availability
                availability = details.get('availability', [])
                for avail in availability:
                    session.run("""
                        MATCH (p:Person {graph_name: $graph_name, id: $node_id})
                        CREATE (p)-[:AVAILABILITY {
                            date: $date,
                            score: $score,
                            content: $content
                        }]->(p)
                    """,
                    graph_name=graph_name,
                    node_id=node_id,
                    date=avail['date'],
                    score=avail['score'],
                    content=avail['content']
                    )

            # Create edges
            for edge in ego_data.edges:
                source, target, edge_data = edge[0], edge[1], edge[2] if len(edge) > 2 else {}

                if isinstance(edge_data, dict):
                    actual = edge_data.get('actual', 0.0)
                    channels = edge_data.get('channels', [])
                else:
                    actual = edge_data if edge_data is not None else 0.0
                    channels = []

                session.run("""
                    MATCH (p1:Person {graph_name: $graph_name, id: $source})
                    MATCH (p2:Person {graph_name: $graph_name, id: $target})
                    CREATE (p1)-[:CONNECTED_TO {
                        actual: $actual,
                        channels: $channels
                    }]->(p2)
                """,
                graph_name=graph_name,
                source=source,
                target=target,
                actual=actual,
                channels=channels
                )

            # Add contact points as Event nodes
            if contact_points:
                for event_type in ['past', 'present', 'potential']:
                    events = contact_points.get(event_type, [])
                    for event in events:
                        event_id = f"{graph_name}_{event_type}_{hash(event['content'])}"
                        session.run("""
                            CREATE (e:Event {
                                graph_name: $graph_name,
                                id: $event_id,
                                type: $event_type,
                                date: $date,
                                content: $content
                            })
                        """,
                        graph_name=graph_name,
                        event_id=event_id,
                        event_type=event_type,
                        date=event.get('date', ''),
                        content=event['content']
                        )

                        people = event.get('people', [])
                        for person_id in people:
                            session.run("""
                                MATCH (e:Event {graph_name: $graph_name, id: $event_id})
                                MATCH (p:Person {graph_name: $graph_name, id: $person_id})
                                CREATE (e)-[:INVOLVES]->(p)
                            """,
                            graph_name=graph_name,
                            event_id=event_id,
                            person_id=person_id
                            )

            # Create vector indexes for similarity search
            _create_vector_indexes(session, graph_name, embedding_dim)
