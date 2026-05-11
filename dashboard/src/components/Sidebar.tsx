import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import axios from 'axios';
import { 
  Activity, 
  Settings, 
  BrainCircuit,
  Terminal
} from 'lucide-react';

import { API_BASE_URL } from '../services/api';

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
    { name: 'Runs', path: '/', icon: Activity },
    { name: 'Dataset', path: '/dataset', icon: BrainCircuit },
    { name: 'Inference', path: '/inference', icon: Terminal },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h2 style={{ fontSize: '1rem', color: 'var(--text-primary)', opacity: 0.9 }}>IN-BED</h2>
        <h2 style={{ fontSize: '1rem', color: 'var(--accent-lime)' }}>POSE</h2>
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
            <item.icon size={22} strokeWidth={isActive ? 2.5 : 2} />
            <span className="nav-label">{item.name}</span>
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="status-indicator">
          <div className={`dot ${isOnline ? 'online' : 'offline'}`} style={{ 
            backgroundColor: isOnline ? 'var(--accent-lime)' : 'var(--accent-pink)',
            boxShadow: isOnline ? '0 0 8px var(--accent-lime)' : '0 0 8px var(--accent-pink)'
          }}></div>
          <span className="micro-label" style={{ 
            color: isOnline ? 'var(--accent-lime)' : 'var(--accent-pink)',
            opacity: 0.8
          }}>
            {isOnline ? 'LIVE' : 'OFF'}
          </span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
