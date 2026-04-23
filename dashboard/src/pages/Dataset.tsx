import React, { useState, useEffect } from 'react';
import {
  Database,
  Filter,
  Maximize2,
  Image as ImageIcon,
  Activity,
  User,
  Layers,
  X
} from 'lucide-react';
import { getDatasetStats, getSamples, getSampleDetail } from '../services/api';

// --- Constants & Skeleton Config ---

const SKELETON_CONNECTIONS = [
  [13, 12], // Head - Thorax
  [12, 8],  // Thorax - R-Shoulder
  [8, 7],   // R-Shoulder - R-Elbow
  [7, 6],   // R-Elbow - R-Wrist
  [12, 9],  // Thorax - L-Shoulder
  [9, 10],  // L-Shoulder - L-Elbow
  [10, 11], // L-Elbow - L-Wrist
  [8, 2],   // R-Shoulder - R-Hip
  [9, 3],   // L-Shoulder - L-Hip
  [2, 3],   // R-Hip - L-Hip
  [2, 1],   // R-Hip - R-Knee
  [1, 0],   // R-Knee - R-Ankle
  [3, 4],   // L-Hip - L-Knee
  [4, 5]    // L-Knee - L-Ankle
];

const JOINT_COLORS: Record<number, string> = {
  13: '#fa7faa', // Head
  12: '#ffb287', // Thorax
  8: '#c2ef4e', 7: '#c2ef4e', 6: '#c2ef4e', // Right arm
  9: '#6a5fc1', 10: '#6a5fc1', 11: '#6a5fc1', // Left arm
  2: '#fa7faa', 1: '#fa7faa', 0: '#fa7faa', // Right leg
  3: '#6a5fc1', 4: '#6a5fc1', 5: '#6a5fc1' // Left leg
};

const JOINT_NAMES = [
  'R-Ankle', 'R-Knee', 'R-Hip', 'L-Hip', 'L-Knee', 'L-Ankle',
  'R-Wrist', 'R-Elbow', 'R-Shoulder', 'L-Shoulder', 'L-Elbow', 'L-Wrist',
  'Thorax', 'Head'
];

// --- Sub-components ---

const JointOverlay = ({ joints, width, height }: { joints: number[][], width: number, height: number }) => {
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none'
      }}
    >
      {/* Lines */}
      {SKELETON_CONNECTIONS.map(([idx1, idx2], i) => {
        const j1 = joints[idx1];
        const j2 = joints[idx2];
        if (!j1 || !j2 || isNaN(j1[0]) || isNaN(j1[1]) || isNaN(j2[0]) || isNaN(j2[1])) return null;
        return (
          <line
            key={`line-${i}`}
            x1={j1[0]} y1={j1[1]}
            x2={j2[0]} y2={j2[1]}
            stroke="white"
            strokeWidth="2"
            strokeOpacity="0.4"
          />
        );
      })}

      {/* Joints */}
      {joints.map((joint, idx) => {
        if (!joint || isNaN(joint[0]) || isNaN(joint[1])) return null;
        return (
          <g key={`joint-${idx}`}>
            <circle
              cx={joint[0]}
              cy={joint[1]}
              r="4"
              fill={JOINT_COLORS[idx] || '#ffffff'}
              stroke="white"
              strokeWidth="1"
            />
            <text
              x={joint[0] + 6}
              y={joint[1] + 4}
              fill="white"
              fontSize="10"
              style={{ textShadow: '1px 1px 2px black' }}
            >
              {JOINT_NAMES[idx]}
            </text>
          </g>
        );
      })}
    </svg>
  );
};

// --- Main Component ---

const Dataset: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [samples, setSamples] = useState<any[]>([]);
  const [split, setSplit] = useState('train');
  const [modality, setModality] = useState('all');
  const [cover, setCover] = useState('all');
  const [selectedSample, setSelectedSample] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
    fetchSamples();
  }, [split, modality, cover]);

  const fetchStats = async () => {
    try {
      const data = await getDatasetStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats', err);
    }
  };

  const fetchSamples = async () => {
    setLoading(true);
    try {
      const data = await getSamples({
        split,
        modality: modality === 'all' ? undefined : modality,
        cover: cover === 'all' ? undefined : cover,
        limit: 12
      });
      setSamples(data.samples || []);
    } catch (err) {
      console.error('Failed to fetch samples', err);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenSample = async (sample: any) => {
    try {
      const detail = await getSampleDetail(split, sample.index);
      setSelectedSample(detail);
    } catch (err) {
      console.error('Failed to fetch sample detail', err);
    }
  };

  return (
    <div className="page-container">
      {/* Local Styles for Dataset Page */}
      <style>{`
        .dataset-header {
          margin-bottom: 32px;
        }
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
          margin-bottom: 40px;
        }
        .stat-card {
          padding: 20px;
          border-radius: 12px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .stat-value {
          font-size: 2rem;
          font-weight: 700;
          font-family: var(--font-display);
        }
        .filters-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
          flex-wrap: wrap;
          gap: 16px;
        }
        .filter-group {
          display: flex;
          gap: 8px;
        }
        .filter-btn {
          padding: 8px 16px;
          border-radius: 8px;
          font-size: 0.85rem;
          font-weight: 600;
          background: var(--bg-secondary);
          color: var(--text-secondary);
          border: 1px solid var(--border-purple);
          cursor: pointer;
        }
        .filter-btn.active {
          background: var(--accent-vibrant);
          color: white;
          border-color: var(--accent-primary);
        }
        
        /* FIXED: Added .select-custom styling */
        .select-custom {
          padding: 8px 32px 8px 16px;
          border-radius: 8px;
          font-size: 0.85rem;
          font-weight: 600;
          background-color: var(--bg-secondary);
          color: var(--text-secondary);
          border: 1px solid var(--border-purple);
          cursor: pointer;
          appearance: none;
          /* Custom SVG arrow */
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23a0a0b0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
          background-repeat: no-repeat;
          background-position: right 12px center;
          background-size: 14px;
        }
        .select-custom:focus {
          outline: none;
          border-color: var(--accent-primary);
        }
        .select-custom option {
          background-color: #1a1a24; /* Dark fallback for dropdown list */
          color: white;
        }

        .samples-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 24px;
        }
        .sample-card {
          border-radius: 12px;
          overflow: hidden;
          transition: transform 0.2s ease;
          cursor: pointer;
        }
        .sample-card:hover {
          transform: translateY(-4px);
        }
        .image-container {
          position: relative;
          aspect-ratio: 4/5;
          background: #111;
        }
        .sample-img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        .sample-info {
          padding: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .sample-meta {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .modality-badge {
          font-size: 0.7rem;
          padding: 2px 6px;
          border-radius: 4px;
          background: var(--accent-primary);
          width: fit-content;
          color: white;
        }

        /* Modal Styles */
        .modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.85);
          backdrop-filter: blur(8px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 20px;
        }
        .modal-container {
          width: 100%;
          max-width: 1100px;
          max-height: 90vh;
          border-radius: 16px;
          overflow: hidden;
          display: grid;
          grid-template-columns: 1fr 320px;
        }
        .modal-main {
          position: relative;
          background: #000;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
        }
        .modal-image-wrapper {
          position: relative;
          max-width: 100%;
          max-height: 100%;
        }
        .modal-image-wrapper img {
          display: block;
          max-width: 100%;
          max-height: 80vh;
          object-fit: contain;
        }
        .modal-sidebar {
          padding: 32px;
          display: flex;
          flex-direction: column;
          gap: 24px;
          border-left: 1px solid var(--border-purple);
          overflow-y: auto;
          background: var(--bg-secondary);
        }
        .close-btn {
          position: absolute;
          top: 20px;
          right: 20px;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--glass-white);
          color: white;
          z-index: 1010;
          border: none;
          cursor: pointer;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      <div className="dataset-header">
        <h1>Dataset Explorer</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Explore SLP Dataset samples and ground-truth annotations.</p>
      </div>

      {/* Stats Summary */}
      <div className="stats-grid">
        <div className="stat-card glass">
          <span className="micro-label">Total Samples</span>
          <span className="stat-value">{stats?.total?.toLocaleString() ?? '---'}</span>
          <Database size={16} color="var(--accent-lime)" />
        </div>
        <div className="stat-card glass">
          <span className="micro-label">Training Set</span>
          <span className="stat-value">{stats?.train ?? '---'}</span>
          <Layers size={16} color="var(--accent-coral)" />
        </div>
        <div className="stat-card glass">
          <span className="micro-label">Validation Set</span>
          <span className="stat-value">{stats?.valid ?? '---'}</span>
          <Activity size={16} color="var(--accent-pink)" />
        </div>
        <div className="stat-card glass">
          <span className="micro-label">Modalities</span>
          <span className="stat-value">{stats?.modalities?.length ?? '---'}</span>
          <ImageIcon size={16} color="var(--accent-primary)" />
        </div>
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <div className="filter-group">
          <button
            className={`filter-btn ${split === 'train' ? 'active' : ''}`}
            onClick={() => setSplit('train')}
          >
            Train
          </button>
          <button
            className={`filter-btn ${split === 'val' ? 'active' : ''}`}
            onClick={() => setSplit('val')}
          >
            Validation
          </button>
        </div>

        <div className="filter-group">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginRight: '16px' }}>
            <Filter size={16} />
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <select
              className="glass select-custom"
              value={modality}
              onChange={(e) => setModality(e.target.value)}
            >
              <option value="all">All Modalities</option>
              {stats?.modalities?.map((m: string) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <select
              className="glass select-custom"
              value={cover}
              onChange={(e) => setCover(e.target.value)}
            >
              <option value="all">All Covers</option>
              {stats?.covers?.map((c: string) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Samples Grid */}
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '100px' }}>
          <Activity className="spin" size={48} color="var(--accent-primary)" />
        </div>
      ) : (
        <div className="samples-grid">
          {samples.map((sample) => (
            <div
              key={sample.id}
              className="sample-card glass"
              onClick={() => handleOpenSample(sample)}
            >
              <div className="image-container">
                <img
                  src={`http://localhost:8000/dataset/image/${split}/${sample.index}`}
                  alt={sample.id}
                  className="sample-img"
                />
                <div style={{ position: 'absolute', top: '12px', right: '12px' }}>
                  <span className="modality-badge">{sample.modality}</span>
                </div>
              </div>
              <div className="sample-info">
                <div className="sample-meta">
                  <span style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{sample.id}</span>
                  <span className="micro-label">Cover: {sample.cover}</span>
                </div>
                <Maximize2 size={16} color="var(--text-secondary)" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      {selectedSample && (
        <div className="modal-overlay" onClick={() => setSelectedSample(null)}>
          <button className="close-btn" onClick={() => setSelectedSample(null)}>
            <X size={24} />
          </button>

          <div className="modal-container glass" onClick={(e) => e.stopPropagation()}>
            <div className="modal-main">
              <div className="modal-image-wrapper">
                <img
                  src={`http://localhost:8000/dataset/image/${split}/${selectedSample.id}`}
                  alt={selectedSample.id}
                />
                {selectedSample.joints && (
                  <JointOverlay
                    joints={selectedSample.joints}
                    width={selectedSample.width}
                    height={selectedSample.height}
                  />
                )}
              </div>
            </div>

            <div className="modal-sidebar glass">
              <div>
                <h2 style={{ marginBottom: '8px' }}>Sample Detail</h2>
                <div className="modality-badge" style={{ marginBottom: '16px' }}>{selectedSample.modality}</div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <User size={18} color="var(--accent-lime)" />
                  <div>
                    <div className="micro-label">Sample ID</div>
                    <div style={{ fontWeight: 600 }}>{selectedSample.id}</div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Layers size={18} color="var(--accent-coral)" />
                  <div>
                    <div className="micro-label">Cover Type</div>
                    <div style={{ fontWeight: 600 }}>{selectedSample.cover}</div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Maximize2 size={18} color="var(--accent-pink)" />
                  <div>
                    <div className="micro-label">Resolution</div>
                    <div style={{ fontWeight: 600 }}>{selectedSample.width} × {selectedSample.height}</div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 'auto', paddingTop: '24px', borderTop: '1px solid var(--border-purple)' }}>
                <div className="micro-label" style={{ marginBottom: '12px' }}>Joint Coordinates</div>
                <div style={{
                  background: 'var(--bg-secondary)',
                  borderRadius: '8px',
                  padding: '12px',
                  fontSize: '0.75rem',
                  maxHeight: '150px',
                  overflowY: 'auto',
                  fontFamily: 'var(--font-mono)'
                }}>
                  {selectedSample.joints ? (
                    selectedSample.joints.map((j: any, i: number) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{JOINT_NAMES[i] || `J${i}`}</span>
                        <span style={{ color: 'var(--accent-lime)' }}>
                          {Array.isArray(j) && j.length >= 2
                            ? `${Math.round(j[0])}, ${Math.round(j[1])}`
                            : 'N/A'}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>No annotations available</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default Dataset;