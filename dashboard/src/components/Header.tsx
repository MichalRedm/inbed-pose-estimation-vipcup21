import React from 'react';
import { Bell, Search, User } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="header glass">
      <div className="header-search">
        <Search size={18} color="var(--text-secondary)" />
        <input type="text" placeholder="Search experiments, models..." />
      </div>
      <div style={{ flex: 1 }}></div>
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
