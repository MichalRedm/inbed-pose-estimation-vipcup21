import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import Training from './pages/Training';
import Inference from './pages/Inference';
import Models from './pages/Models';
import Settings from './pages/Settings';

// Placeholder components for remaining pages
const Dataset = () => <div><h1 className="text-uppercase">Dataset Explorer</h1></div>;



const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="training" element={<Training />} />
          <Route path="inference" element={<Inference />} />
          <Route path="models" element={<Models />} />
          <Route path="dataset" element={<Dataset />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
