import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import Settings from './pages/Settings';
import Dataset from './pages/Dataset';
import Augmentations from './pages/Augmentations';
import Inference from './pages/Inference';
import { GlobalStateProvider } from './context/GlobalStateContext';

const App: React.FC = () => {
  return (
    <GlobalStateProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="dataset" element={<Dataset />} />
            <Route path="augmentations" element={<Augmentations />} />
            <Route path="inference" element={<Inference />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </Router>
    </GlobalStateProvider>
  );
};

export default App;
