import React, { useState, useEffect } from 'react';
import { getDatasetStats, getSamples, getDatasetImageUrl, getSampleDetail } from '../services/api';
import { 
  Database, 
  Image as ImageIcon, 
  Users, 
  Layers, 
  Filter, 
  ChevronLeft, 
  ChevronRight, 
  X
} from 'lucide-react';

interface DatasetStats {
  train: {
    total: number;
    modalities: Record<string, number>;
    covers: Record<string, number>;
    subject_count: number;
  };
  val: {
    total: number;
    modalities: Record<string, number>;
    covers: Record<string, number>;
    subject_count: number;
  };
}

interface Sample {
  id: number;
  subject: number;
  modality: string;
  cover: string;
  filename: string;
  has_joints: boolean;
}

interface Joint {
  name: string;
  x: number;
  y: number;
  visible: boolean;
}

interface SampleDetail extends Sample {
  joints?: Joint[];
  split: string;
}

const Dataset: React.FC = () => {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [totalSamples, setTotalSamples] = useState(0);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [split, setSplit] = useState('train');
  const [modality, setModality] = useState<string>('');
  const [cover, setCover] = useState<string>('');
  const [selectedSample, setSelectedSample] = useState<SampleDetail | null>(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const loadStats = async () => {
      try {
        const data = await getDatasetStats();
        if (isMounted) setStats(data);
      } catch (error) {
        console.error('Failed to fetch dataset stats:', error);
      }
    };
    loadStats();
    return () => { isMounted = false; };
  }, []);

  useEffect(() => {
    let isMounted = true;
    const loadSamples = async () => {
      setLoading(true);
      try {
        const data = await getSamples({
          split,
          page,
          limit: 12,
          modality: modality || undefined,
          cover: cover || undefined,
        });
        if (isMounted) {
          setSamples(data.samples);
          setTotalSamples(data.total);
        }
      } catch (error) {
        console.error('Failed to fetch samples:', error);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    loadSamples();
    return () => { isMounted = false; };
  }, [split, page, modality, cover]);

  const handleSampleClick = async (sample: Sample) => {
    try {
      const detail = await getSampleDetail(split, sample.id);
      setSelectedSample(detail);
      setShowModal(true);
    } catch (error) {
      console.error('Failed to fetch sample detail:', error);
    }
  };

  const totalPages = Math.ceil(totalSamples / 12);

  return (
    <div className="dataset-page">
      <div className="page-header" style={{ marginBottom: '32px' }}>
        <h1 className="text-uppercase" style={{ marginBottom: '8px' }}>Dataset Explorer</h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Browse and visualize the Simultaneously-collected Multimodal Lying Pose (SLP) dataset.
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid-container" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        <div className="glass card" style={{ padding: '20px', borderRadius: '12px', borderLeft: '4px solid var(--accent-primary)' }}>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(106, 95, 193, 0.2)' }}>
              <Database size={20} color="var(--accent-primary)" />
            </div>
            <div>
              <div className="micro-label text-uppercase">Total Samples</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '700' }}>
                {stats ? stats.train.total + stats.val.total : '...'}
              </div>
            </div>
          </div>
        </div>

        <div className="glass card" style={{ padding: '20px', borderRadius: '12px', borderLeft: '4px solid var(--accent-lime)' }}>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(194, 239, 78, 0.2)' }}>
              <Users size={20} color="var(--accent-lime)" />
            </div>
            <div>
              <div className="micro-label text-uppercase">Subjects</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '700' }}>
                {stats ? stats.train.subject_count + stats.val.subject_count : '...'}
              </div>
            </div>
          </div>
        </div>

        <div className="glass card" style={{ padding: '20px', borderRadius: '12px', borderLeft: '4px solid var(--accent-pink)' }}>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(250, 127, 170, 0.2)' }}>
              <Layers size={20} color="var(--accent-pink)" />
            </div>
            <div>
              <div className="micro-label text-uppercase">Modalities</div>
              <div style={{ fontSize: '1.1rem', fontWeight: '600' }}>
                {stats ? Object.keys(stats.train.modalities).join(', ') : '...'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Browser Section */}
      <div className="glass" style={{ borderRadius: '16px', overflow: 'hidden', border: '1px solid var(--border-purple)' }}>
        {/* Filters */}
        <div style={{ padding: '20px', borderBottom: '1px solid var(--border-purple)', background: 'rgba(255,255,255,0.02)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '20px' }}>
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <div style={{ display: 'flex', background: 'var(--bg-secondary)', borderRadius: '8px', padding: '4px' }}>
              <button 
                onClick={() => {setSplit('train'); setPage(1);}}
                className={split === 'train' ? 'active-tab' : 'inactive-tab'}
                style={{ padding: '6px 16px', borderRadius: '6px', fontSize: '0.85rem' }}
              >
                Train
              </button>
              <button 
                onClick={() => {setSplit('val'); setPage(1);}}
                className={split === 'val' ? 'active-tab' : 'inactive-tab'}
                style={{ padding: '6px 16px', borderRadius: '6px', fontSize: '0.85rem' }}
              >
                Validation
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <Filter size={16} color="var(--text-secondary)" />
              <select 
                value={modality} 
                onChange={(e) => {setModality(e.target.value); setPage(1);}}
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-purple)', color: 'white', padding: '6px 12px', borderRadius: '8px', fontSize: '0.85rem' }}
              >
                <option value="">All Modalities</option>
                <option value="RGB">RGB</option>
                <option value="IR">IR</option>
              </select>
              <select 
                value={cover} 
                onChange={(e) => {setCover(e.target.value); setPage(1);}}
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-purple)', color: 'white', padding: '6px 12px', borderRadius: '8px', fontSize: '0.85rem' }}
              >
                <option value="">All Covers</option>
                <option value="uncover">Uncover</option>
                <option value="cover1">Cover 1</option>
                <option value="cover2">Cover 2</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Page {page} of {totalPages || 1} ({totalSamples} samples)
            </span>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button 
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
                className="icon-btn glass" 
                style={{ width: '32px', height: '32px', borderRadius: '6px', opacity: page === 1 ? 0.5 : 1 }}
              >
                <ChevronLeft size={18} />
              </button>
              <button 
                disabled={page === totalPages || totalPages === 0}
                onClick={() => setPage(p => p + 1)}
                className="icon-btn glass" 
                style={{ width: '32px', height: '32px', borderRadius: '6px', opacity: (page === totalPages || totalPages === 0) ? 0.5 : 1 }}
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* Grid */}
        <div style={{ padding: '24px', minHeight: '600px' }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', flexDirection: 'column', gap: '16px' }}>
              <div className="spin" style={{ width: '40px', height: '40px', border: '4px solid var(--glass-white)', borderTopColor: 'var(--accent-primary)', borderRadius: '50%' }}></div>
              <p style={{ color: 'var(--text-secondary)' }}>Loading samples...</p>
            </div>
          ) : samples.length > 0 ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '20px' }}>
              {samples.map((sample) => (
                <div 
                  key={sample.id} 
                  className="sample-card"
                  onClick={() => handleSampleClick(sample)}
                  style={{ 
                    background: 'var(--bg-secondary)', 
                    borderRadius: '12px', 
                    overflow: 'hidden', 
                    border: '1px solid var(--border-purple)',
                    cursor: 'pointer',
                    transition: 'transform 0.2s ease, border-color 0.2s ease'
                  }}
                >
                  <div style={{ height: '180px', background: '#000', position: 'relative', overflow: 'hidden' }}>
                    <img 
                      src={getDatasetImageUrl(split, sample.id)} 
                      alt={sample.filename}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      loading="lazy"
                    />
                    <div style={{ position: 'absolute', top: '8px', right: '8px' }}>
                      <span style={{ 
                        padding: '4px 8px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem', 
                        fontWeight: '700',
                        background: sample.modality === 'RGB' ? 'var(--accent-primary)' : 'var(--accent-pink)',
                        color: 'white',
                        textTransform: 'uppercase'
                      }}>
                        {sample.modality}
                      </span>
                    </div>
                  </div>
                  <div style={{ padding: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: '600' }}>Subject {sample.subject}</span>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{sample.cover}</span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {sample.filename}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px', flexDirection: 'column', gap: '16px', color: 'var(--text-secondary)' }}>
              <ImageIcon size={48} opacity={0.3} />
              <p>No samples found matching the filters.</p>
            </div>
          )}
        </div>
      </div>

      {/* Modal */}
      {showModal && selectedSample && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px' }}>
          <div className="glass" style={{ width: '100%', maxWidth: '1000px', maxHeight: '90vh', borderRadius: '20px', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-purple)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2 style={{ fontSize: '1.2rem' }}>Sample Details</h2>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{selectedSample.filename}</p>
              </div>
              <button onClick={() => setShowModal(false)} className="icon-btn glass" style={{ width: '36px', height: '36px', borderRadius: '50%' }}>
                <X size={20} />
              </button>
            </div>
            
            <div style={{ flex: 1, overflow: 'auto', display: 'grid', gridTemplateColumns: '1fr 300px', gap: '0' }}>
              <div style={{ background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', minHeight: '500px' }}>
                <img 
                  id="modal-image"
                  src={getDatasetImageUrl(selectedSample.split, selectedSample.id)} 
                  alt="Detail"
                  style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                />
                {/* Joints Overlay */}
                <svg 
                  viewBox="0 0 256 256" 
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                  preserveAspectRatio="xMidYMid meet"
                >
                  {selectedSample.joints?.map((joint, i) => (
                    joint.visible && (
                      <g key={i}>
                        <circle cx={joint.x} cy={joint.y} r="3" fill="var(--accent-lime)" />
                        <text x={joint.x + 5} y={joint.y} fill="white" fontSize="6" style={{ textShadow: '0 0 2px black' }}>{joint.name}</text>
                      </g>
                    )
                  ))}
                </svg>
              </div>
              
              <div style={{ padding: '24px', background: 'var(--bg-secondary)', borderLeft: '1px solid var(--border-purple)' }}>
                <h3 className="text-uppercase micro-label" style={{ marginBottom: '16px', color: 'var(--accent-primary)' }}>Metadata</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '32px' }}>
                  <div>
                    <div className="micro-label">Subject</div>
                    <div style={{ fontWeight: '600' }}>{selectedSample.subject}</div>
                  </div>
                  <div>
                    <div className="micro-label">Split</div>
                    <div style={{ fontWeight: '600', textTransform: 'capitalize' }}>{selectedSample.split}</div>
                  </div>
                  <div>
                    <div className="micro-label">Modality</div>
                    <div style={{ fontWeight: '600' }}>{selectedSample.modality}</div>
                  </div>
                  <div>
                    <div className="micro-label">Cover</div>
                    <div style={{ fontWeight: '600', textTransform: 'capitalize' }}>{selectedSample.cover}</div>
                  </div>
                </div>

                <h3 className="text-uppercase micro-label" style={{ marginBottom: '16px', color: 'var(--accent-lime)' }}>Joints ({selectedSample.joints?.length || 0})</h3>
                <div style={{ maxHeight: '300px', overflowY: 'auto', paddingRight: '8px' }}>
                   {selectedSample.joints ? (
                     <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                       <tbody>
                         {selectedSample.joints.map((j, i) => (
                           <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                             <td style={{ padding: '6px 0', color: 'var(--text-secondary)' }}>{j.name}</td>
                             <td style={{ padding: '6px 0', textAlign: 'right' }}>
                               <span style={{ color: j.visible ? 'var(--accent-lime)' : '#666' }}>
                                 {j.visible ? `(${Math.round(j.x)}, ${Math.round(j.y)})` : 'Hidden'}
                               </span>
                             </td>
                           </tr>
                         ))}
                       </tbody>
                     </table>
                   ) : (
                     <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>No annotations available.</p>
                   )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .active-tab {
          background: var(--accent-vibrant);
          color: white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .inactive-tab {
          background: transparent;
          color: var(--text-secondary);
        }
        .inactive-tab:hover {
          color: white;
          background: rgba(255,255,255,0.05);
        }
        .sample-card:hover {
          transform: translateY(-4px);
          border-color: var(--accent-primary) !important;
          box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }
        select:focus {
          border-color: var(--accent-primary) !important;
          outline: none;
        }
      `}</style>
    </div>
  );
};

export default Dataset;
