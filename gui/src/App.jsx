import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { EgoGraphView } from './components/EgoGraphView';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<EgoGraphView />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
