import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Activity, 
  Eye, 
  Settings, 
  Database, 
  Box 
} from 'lucide-react';

const Sidebar: React.FC = () => {
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
          <div className="dot online"></div>
          <span className="micro-label text-uppercase">Backend Online</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
