import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Overview from './pages/Overview';

// Placeholder components for other pages
const Training = () => <div><h1 className="text-uppercase">Training Monitor</h1></div>;
const Inference = () => <div><h1 className="text-uppercase">Model Inference</h1></div>;
const Models = () => <div><h1 className="text-uppercase">Model Management</h1></div>;
const Dataset = () => <div><h1 className="text-uppercase">Dataset Explorer</h1></div>;
const Settings = () => <div><h1 className="text-uppercase">Settings</h1></div>;

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
