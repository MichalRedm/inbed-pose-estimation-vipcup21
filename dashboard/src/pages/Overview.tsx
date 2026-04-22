import React from 'react';

const Overview: React.FC = () => {
  return (
    <div>
      <h1 className="text-uppercase" style={{ marginBottom: '24px' }}>Dashboard Overview</h1>
      <div className="grid-container" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
        <div className="glass card" style={{ padding: '24px', borderRadius: '12px' }}>
          <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-lime)' }}>Latest Experiment</h3>
          <h2 style={{ margin: '8px 0' }}>HRNet-W32-Base</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Status: Completed</p>
          <div style={{ marginTop: '16px' }}>
            <span style={{ fontSize: '2rem', fontWeight: '700' }}>84.2%</span>
            <span style={{ marginLeft: '8px', color: 'var(--accent-lime)' }}>mAP</span>
          </div>
        </div>
        <div className="glass card" style={{ padding: '24px', borderRadius: '12px' }}>
          <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-pink)' }}>System Status</h3>
          <h2 style={{ margin: '8px 0' }}>NVIDIA RTX 3090</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Memory: 12.4 / 24 GB</p>
          <div style={{ marginTop: '16px', height: '8px', background: 'var(--bg-secondary)', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ width: '52%', height: '100%', background: 'var(--accent-pink)' }}></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Overview;
