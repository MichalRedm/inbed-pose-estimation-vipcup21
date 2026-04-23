import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import Training from './pages/Training';
import Inference from './pages/Inference';
import Models from './pages/Models';
import Settings from './pages/Settings';
import Dataset from './pages/Dataset';
import Evaluation from './pages/Evaluation';



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
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
