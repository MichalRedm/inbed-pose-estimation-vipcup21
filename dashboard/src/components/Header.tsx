import React from 'react';
import { Bell, Search, User, Database } from 'lucide-react';
import GlobalModelSelector from './GlobalModelSelector';

const Header: React.FC = () => {
  return (
    <header className="header glass">
      <div className="header-search">
        <Search size={18} color="var(--text-secondary)" />
        <input type="text" placeholder="Search experiments, models..." />
      </div>
      <div style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Database size={16} color="var(--accent-pink)" />
          <span className="micro-label text-uppercase" style={{ opacity: 0.7 }}>Active Model:</span>
          <GlobalModelSelector />
        </div>
      </div>
      <div className="header-actions">
        <button className="icon-btn">
          <Bell size={20} />
        </button>
        <div className="user-profile">
          <div className="user-avatar">
            <User size={20} />
          </div>
          <span className="user-name">Developer</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
