import { BrowserRouter, Routes, Route, Link, useNavigate } from 'react-router-dom';
import { EgoGraphView } from './components/EgoGraphView';
import './App.css';

function HomePage() {
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const graphName = formData.get('graphName');
    if (graphName) {
      navigate(`/${graphName}`);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
      <div className="max-w-2xl mx-auto px-6 py-12">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          Drop in a Pond
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          A semantic network navigation system for understanding and strategically
          navigating your social and professional networks.
        </p>

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            View an Ego Graph
          </h2>
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              name="graphName"
              placeholder="Enter graph name (e.g., fronx)"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              defaultValue="fronx"
            />
            <button
              type="submit"
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Load
            </button>
          </form>
          <p className="text-xs text-gray-500 mt-3">
            Loads from <code>/data/ego_graphs/[name].json</code>
          </p>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">Quick Start</h3>
          <ul className="space-y-1 text-sm text-blue-800">
            <li>
              <Link to="/fronx" className="underline hover:text-blue-600">
                View example: fronx
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/:graphName" element={<EgoGraphView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
