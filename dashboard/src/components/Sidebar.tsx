import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import axios from 'axios';
import { 
  LayoutDashboard, 
  Activity, 
  Eye, 
  Settings, 
  Database, 
  Box 
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const Sidebar: React.FC = () => {
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        await axios.get(API_BASE_URL);
        setIsOnline(true);
      } catch {
        setIsOnline(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { name: 'Overview', path: '/', icon: LayoutDashboard },
    { name: 'Training', path: '/training', icon: Activity },
    { name: 'Inference', path: '/inference', icon: Eye },
    { name: 'Models', path: '/models', icon: Box },
    { name: 'Dataset', path: '/dataset', icon: Database },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h2 className="text-uppercase" style={{ fontSize: '1.2rem', color: 'var(--accent-lime)' }}>In-Bed Pose</h2>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            <item.icon size={20} />
            <span className="text-uppercase">{item.name}</span>
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="status-indicator">
          <div className={`dot ${isOnline ? 'online' : 'offline'}`} style={{ 
            backgroundColor: isOnline ? 'var(--accent-lime)' : 'var(--accent-pink)',
            boxShadow: isOnline ? '0 0 8px var(--accent-lime)' : '0 0 8px var(--accent-pink)'
          }}></div>
          <span className="micro-label text-uppercase" style={{ color: isOnline ? 'var(--text-primary)' : 'var(--accent-pink)' }}>
            {isOnline ? 'Backend Online' : 'Backend Offline'}
          </span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
