import React, { useState, useEffect } from 'react';
import {
  Wand2,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  Dices,
  Image as ImageIcon,
  AlertCircle,
  Settings2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getSamples, getAvailableAugmentations, applyAugmentations } from '../services/api';

// --- Constants & Helper Components ---

const SKELETON_CONNECTIONS = [
  [13, 12], [12, 8], [8, 7], [7, 6], [12, 9], [9, 10], [10, 11],
  [8, 2], [9, 3], [2, 3], [2, 1], [1, 0], [3, 4], [4, 5]
];

const JOINT_COLORS: Record<number, string> = {
  13: '#fa7faa', 12: '#ffb287',
  8: '#c2ef4e', 7: '#c2ef4e', 6: '#c2ef4e',
  9: '#6a5fc1', 10: '#6a5fc1', 11: '#6a5fc1',
  2: '#fa7faa', 1: '#fa7faa', 0: '#fa7faa',
  3: '#6a5fc1', 4: '#6a5fc1', 5: '#6a5fc1'
};

const JointOverlay = ({ joints, width, height }: { joints: Array<{x: number, y: number}>, width: number, height: number }) => {
  const scale = Math.min(width, height) / 256;
  const dotRadius = Math.max(1, 4 * scale);
  const strokeWidth = Math.max(0.5, 2 * scale);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      {SKELETON_CONNECTIONS.map(([idx1, idx2], i) => {
        const j1 = joints[idx1];
        const j2 = joints[idx2];
        if (!j1 || !j2) return null;
        return (
          <line
            key={`line-${i}`}
            x1={j1.x} y1={j1.y}
            x2={j2.x} y2={j2.y}
            stroke="white"
            strokeWidth={strokeWidth}
            strokeOpacity="0.4"
          />
        );
      })}

      {joints.map((joint, idx) => (
        <circle
          key={`joint-${idx}`}
          cx={joint.x}
          cy={joint.y}
          r={dotRadius}
          fill={JOINT_COLORS[idx] || '#ffffff'}
          stroke="white"
          strokeWidth={strokeWidth / 2}
        />
      ))}
    </svg>
  );
};

// --- Types ---

interface AugmentationParam {
  type: 'float' | 'int' | 'bool' | 'choice';
  min?: number;
  max?: number;
  default: string | number | boolean;
  options?: string[];
}

interface AugmentationMetadata {
  id: string;
  name: string;
  order: number;
  params: Record<string, AugmentationParam>;
}

interface SelectedAugmentation {
  id: string;
  enabled: boolean;
  params: Record<string, string | number | boolean>;
  randomParams: Record<string, boolean>;
}

interface DatasetSample {
  id: string;
  index: number;
}

interface InferenceResult {
  image: string;
  joints: Array<{x: number, y: number}>;
  original_size: {width: number, height: number};
}

// --- Main Component ---

const Augmentations: React.FC = () => {
  const [availableAugs, setAvailableAugs] = useState<AugmentationMetadata[]>([]);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);
  const [modality, setModality] = useState('IR');
  const [augStates, setAugStates] = useState<Record<string, SelectedAugmentation>>({});
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedAug, setExpandedAug] = useState<string | null>(null);

  // Initialize
  useEffect(() => {
    const init = async () => {
      try {
        const [augsData, samplesData] = await Promise.all([
          getAvailableAugmentations(),
          getSamples({ limit: 12, split: 'train' })
        ]);
        
        setAvailableAugs(augsData.augmentations);
        setSamples(samplesData.samples);

        // Build initial states
        const initialStates: Record<string, SelectedAugmentation> = {};
        augsData.augmentations.forEach((aug: AugmentationMetadata) => {
          const params: Record<string, string | number | boolean> = {};
          const randomParams: Record<string, boolean> = {};
          Object.entries(aug.params).forEach(([pName, pMeta]) => {
            params[pName] = pMeta.default;
            randomParams[pName] = true; // Default to random for everything
          });
          initialStates[aug.id] = {
            id: aug.id,
            enabled: false,
            params,
            randomParams
          };
        });
        setAugStates(initialStates);
      } catch (err) {
        console.error('Failed to initialize', err);
        setError('Failed to load augmentations or samples.');
      }
    };
    init();
  }, []);

  const handleApply = async () => {
    if (!samples[selectedSampleIdx]) return;
    setLoading(true);
    setError(null);

    const activeAugs = availableAugs
      .filter(aug => augStates[aug.id].enabled)
      .map(aug => {
        const state = augStates[aug.id];
        const params: Record<string, string | number | boolean> = {};
        Object.entries(state.params).forEach(([pName, pVal]) => {
          if (!state.randomParams[pName]) {
            params[pName] = pVal;
          }
        });
        return { id: aug.id, params };
      });

    try {
      const data = await applyAugmentations({
        split: 'train',
        index: samples[selectedSampleIdx].index,
        modality,
        augmentations: activeAugs
      });
      setResult(data as InferenceResult);
    } catch (err) {
      console.error('Apply failed', err);
      setError('Failed to apply augmentations. Check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const updateParam = (augId: string, paramName: string, value: string | number | boolean) => {
    setAugStates(prev => ({
      ...prev,
      [augId]: {
        ...prev[augId],
        params: { ...prev[augId].params, [paramName]: value },
        randomParams: { ...prev[augId].randomParams, [paramName]: false }
      }
    }));
  };


  const toggleRandom = (augId: string, paramName: string) => {
    setAugStates(prev => ({
      ...prev,
      [augId]: {
        ...prev[augId],
        randomParams: { ...prev[augId].randomParams, [paramName]: !prev[augId].randomParams[paramName] }
      }
    }));
  };

  const toggleAug = (augId: string) => {
    setAugStates(prev => ({
      ...prev,
      [augId]: { ...prev[augId], enabled: !prev[augId].enabled }
    }));
  };

  return (
    <div className="augmentations-page padded-container">
      <div className="page-header">
        <h1 className="text-uppercase">Augmentation Visualizer</h1>
        <p className="text-secondary">Simulate training pipeline augmentations with real-time feedback.</p>
      </div>

      <div className="augmentation-grid" style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 400px', gap: '24px', marginTop: '32px', alignItems: 'start' }}>
        
        {/* Left Panel: Preview */}
        <div className="flex-column" style={{ gap: '24px', minWidth: 0 }}>
          
          <div className="glass card" style={{ padding: '20px', display: 'flex', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: '16px', minWidth: 0 }}>
             <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flex: 1, minWidth: 0 }}>
               <span className="text-uppercase micro-label text-secondary">Sample:</span>
               <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '4px', flex: 1, minWidth: 0 }}>
                 {samples.map((s, i) => (
                   <button 
                    key={s.id} 
                    onClick={() => setSelectedSampleIdx(i)}
                    className={`btn-tab ${selectedSampleIdx === i ? 'active' : ''}`}
                    style={{ padding: '6px 12px', fontSize: '0.8rem', whiteSpace: 'nowrap' }}
                   >
                     {s.id}
                   </button>
                 ))}
               </div>
             </div>
             <div style={{ display: 'flex', gap: '12px', alignItems: 'center', borderLeft: '1px solid var(--border-purple)', paddingLeft: '16px', marginLeft: '16px' }}>
               <span className="text-uppercase micro-label text-secondary">Modality:</span>
               <select 
                value={modality} 
                onChange={(e) => setModality(e.target.value)}
                className="glass-input"
                style={{ padding: '6px 12px', fontSize: '0.85rem', width: 'auto', minWidth: '80px', height: 'auto' }}
               >
                 <option value="IR">IR</option>
                 <option value="RGB">RGB</option>
               </select>
             </div>
          </div>

          <div className="glass card" style={{ padding: 0, height: '500px', position: 'relative', overflow: 'hidden', background: 'rgba(0,0,0,0.2)', border: '2px dashed var(--border-purple)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {loading && (
              <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(2px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
                <RefreshCw className="spin" size={48} color="var(--accent-lime)" />
              </div>
            )}
            
            {result ? (
              <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
                <img 
                  src={result.image} 
                  style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: '8px' }} 
                  alt="Augmented" 
                />
                <JointOverlay joints={result.joints} width={result.original_size.width} height={result.original_size.height} />
              </div>
            ) : (
              <div className="empty-state" style={{ border: 'none' }}>
                <ImageIcon size={64} style={{ opacity: 0.2, marginBottom: '16px' }} />
                <p className="text-secondary">Configure augmentations and click Apply to see results.</p>
              </div>
            )}

            {error && (
              <div style={{ position: 'absolute', bottom: '20px', left: '20px', right: '20px', padding: '12px 16px', background: 'rgba(255, 61, 113, 0.15)', border: '1px solid var(--accent-pink)', borderRadius: '8px', color: 'var(--accent-pink)', display: 'flex', alignItems: 'center', gap: '12px', backdropFilter: 'blur(4px)' }}>
                <AlertCircle size={20} />
                <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>{error}</span>
              </div>
            )}
          </div>

        </div>

        {/* Right Panel: Controls */}
        <div className="flex-column" style={{ gap: '24px' }}>
          <div className="glass card">
            <div className="card-header" style={{ marginBottom: '20px' }}>
              <h3 className="text-uppercase micro-label" style={{ color: 'var(--accent-lime)' }}>Pipeline Config</h3>
              <Settings2 size={18} color="var(--accent-lime)" />
            </div>

            <div className="flex-column" style={{ gap: '12px' }}>
              {availableAugs.map(aug => (
                <div key={aug.id} className="glass" style={{ borderRadius: '12px', border: `1px solid ${augStates[aug.id]?.enabled ? 'var(--accent-primary)' : 'rgba(255, 255, 255, 0.1)'}`, overflow: 'hidden', transition: 'all 0.2s ease' }}>
                  <div 
                    style={{ padding: '14px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', background: augStates[aug.id]?.enabled ? 'rgba(106, 95, 193, 0.15)' : 'rgba(255, 255, 255, 0.02)' }}
                    onClick={() => setExpandedAug(expandedAug === aug.id ? null : aug.id)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <label className="switch-sm" onClick={e => e.stopPropagation()}>
                        <input 
                          type="checkbox" 
                          checked={augStates[aug.id]?.enabled || false} 
                          onChange={() => toggleAug(aug.id)}
                        />
                        <span className="slider"></span>
                      </label>
                      <span style={{ fontWeight: 600, fontSize: '0.95rem', color: augStates[aug.id]?.enabled ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{aug.name}</span>
                    </div>
                    <div style={{ color: 'var(--text-secondary)' }}>
                      {expandedAug === aug.id ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                    </div>
                  </div>

                  <AnimatePresence>
                    {expandedAug === aug.id && (
                      <motion.div 
                        initial={{ height: 0, opacity: 0 }} 
                        animate={{ height: 'auto', opacity: 1 }} 
                        exit={{ height: 0, opacity: 0 }}
                        style={{ overflow: 'hidden' }}
                      >
                        <div style={{ padding: '20px 16px', borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', gap: '20px', background: 'rgba(0,0,0,0.2)' }}>
                          {Object.entries(aug.params).map(([pName, pMeta]) => (
                            <div key={pName} className="param-row">
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                <span className="micro-label" style={{ opacity: 0.8 }}>{pName.replace(/_/g, ' ')}</span>
                                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                                  {!augStates[aug.id].randomParams[pName] && (
                                    <span style={{ color: 'var(--accent-lime)', fontSize: '0.85rem', fontWeight: 600, background: 'rgba(194, 239, 78, 0.1)', padding: '2px 8px', borderRadius: '4px' }}>
                                      {typeof augStates[aug.id].params[pName] === 'number' ? (augStates[aug.id].params[pName] as number).toFixed(2) : String(augStates[aug.id].params[pName])}
                                    </span>
                                  )}
                                  <button 
                                    onClick={() => toggleRandom(aug.id, pName)}
                                    className={`icon-btn ${augStates[aug.id].randomParams[pName] ? 'text-accent-lime active-random' : ''}`}
                                    title="Toggle Randomization"
                                    style={{ padding: '6px', borderRadius: '6px', background: augStates[aug.id].randomParams[pName] ? 'rgba(194, 239, 78, 0.1)' : 'rgba(255,255,255,0.05)' }}
                                  >
                                    <Dices size={14} />
                                  </button>
                                </div>
                              </div>
                              
                              {!augStates[aug.id].randomParams[pName] ? (
                                pMeta.type === 'float' || pMeta.type === 'int' ? (
                                  <div style={{ padding: '0 4px' }}>
                                    <input 
                                      type="range" 
                                      className="custom-range"
                                      min={pMeta.min} 
                                      max={pMeta.max} 
                                      step={pMeta.type === 'float' ? 0.01 : 1}
                                      value={augStates[aug.id].params[pName] as string | number}
                                      onChange={(e) => updateParam(aug.id, pName, parseFloat(e.target.value))}
                                      style={{ width: '100%' }}
                                    />
                                  </div>
                                ) : pMeta.type === 'bool' ? (
                                  <label className="switch-sm">
                                    <input 
                                      type="checkbox" 
                                      checked={augStates[aug.id].params[pName] as boolean} 
                                      onChange={(e) => updateParam(aug.id, pName, e.target.checked)}
                                    />
                                    <span className="slider"></span>
                                  </label>
                                ) : null
                              ) : (
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', opacity: 0.6, display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 12px', background: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
                                  <Dices size={14} /> <span>Using random value during application</span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}
            </div>

            <button 
              className="btn-lime" 
              style={{ width: '100%', marginTop: '24px', padding: '14px', fontSize: '0.95rem' }} 
              onClick={handleApply}
              disabled={loading}
            >
              {loading ? <RefreshCw className="spin" size={18} /> : <Wand2 size={18} />}
              Apply Augmentations
            </button>
          </div>

          <div className="glass card">
            <h3 className="text-uppercase micro-label" style={{ marginBottom: '16px', color: 'var(--text-secondary)' }}>Quick Actions</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <button className="btn-tab" onClick={() => {
                const newState = { ...augStates };
                Object.keys(newState).forEach(id => newState[id].enabled = true);
                setAugStates(newState);
              }} style={{ justifyContent: 'center' }}>Enable All</button>
              <button className="btn-tab" onClick={() => {
                const newState = { ...augStates };
                Object.keys(newState).forEach(id => newState[id].enabled = false);
                setAugStates(newState);
              }} style={{ justifyContent: 'center' }}>Disable All</button>
            </div>
          </div>
        </div>

      </div>

      <style>{`
        .augmentations-page {
          height: 100%;
          overflow-y: auto;
        }
        .param-row {
          display: flex;
          flex-direction: column;
        }
        .icon-btn {
          background: none;
          border: none;
          color: var(--text-secondary);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
          cursor: pointer;
        }
        .icon-btn:hover { color: var(--text-primary); background: rgba(255,255,255,0.1) !important; }
        .icon-btn.active-random { color: var(--accent-lime); border: 1px solid rgba(194, 239, 78, 0.3); }
        
        /* Custom Range Slider */
        .custom-range {
          -webkit-appearance: none;
          width: 100%;
          height: 4px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
          outline: none;
        }
        .custom-range::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--accent-lime);
          cursor: pointer;
          transition: transform 0.1s;
        }
        .custom-range::-webkit-slider-thumb:hover {
          transform: scale(1.2);
        }
      `}</style>
    </div>
  );
};

export default Augmentations;
