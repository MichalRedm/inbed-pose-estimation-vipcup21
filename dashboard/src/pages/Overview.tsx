import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  History, 
  Trash2, 
  Activity, 
  Target, 
  Eye, 
  Info,
  Clock,
  Plus
} from 'lucide-react';
import { getRuns, getRunDetails, deleteRun, getTrainingStatus } from '../services/api';
import { useGlobalState } from '../context/GlobalStateContext';

// Components
import TrainingForm from '../components/TrainingForm';
import RunInference from '../components/RunInference';
import RunAnalysis from '../components/RunAnalysis';

const Overview: React.FC = () => {
  const { selectedRun, setSelectedRun } = useGlobalState();
  const [runs, setRuns] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<'analysis' | 'inference'>('analysis');
  const [isStartingNew, setIsStartingNew] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<any>(null);
  const [runDetails, setRunDetails] = useState<any>(null);

  const fetchRuns = useCallback(async () => {
    try {
      const data = await getRuns();
      setRuns(data.runs);
    } catch (error) {
      console.error('Failed to fetch runs:', error);
    }
  }, []);

  const fetchTrainingStatus = useCallback(async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      
      // If a run just started and we aren't viewing anything, select it
      if (status.is_running && status.run_id && !selectedRun && !isStartingNew) {
        setSelectedRun(status.run_id);
      }
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  }, [selectedRun, isStartingNew, setSelectedRun]);

  useEffect(() => {
    fetchRuns();
    fetchTrainingStatus();
    const interval = setInterval(() => {
      fetchRuns();
      fetchTrainingStatus();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchRuns, fetchTrainingStatus]);

  useEffect(() => {
    if (selectedRun) {
      getRunDetails(selectedRun).then(details => {
        setRunDetails(details);
      });
    } else {
      setRunDetails(null);
    }
  }, [selectedRun]);

  const handleDeleteRun = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm(`Delete run ${id}?`)) return;
    try {
      await deleteRun(id);
      if (selectedRun === id) setSelectedRun('');
      fetchRuns();
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  return (
    <div className="runs-hub-container" style={{ 
      display: 'flex', 
      height: 'calc(100vh - 72px)',
      width: '100%',
      overflow: 'hidden'
    }}>
      
      {/* Runs Browser (Left) */}
      <div className="runs-browser" style={{ 
        width: '320px', 
        background: 'var(--bg-secondary)', 
        borderRight: '1px solid var(--border-purple)', 
        display: 'flex', 
        flexDirection: 'column',
        height: '100%'
      }}>
        <div style={{ padding: '24px', borderBottom: '1px solid var(--border-purple)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 className="text-uppercase" style={{ fontSize: '0.9rem', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
            <History size={16} />
            Runs
          </h2>
          <button 
            className="btn-lime" 
            style={{ padding: '6px', borderRadius: '50%' }}
            onClick={() => { setIsStartingNew(true); setSelectedRun(''); }}
          >
            <Plus size={18} />
          </button>
        </div>

        <div className="runs-list" style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
          {runs.map((run) => (
            <motion.div 
              key={run.id}
              layoutId={run.id}
              onClick={() => { setSelectedRun(run.id); setIsStartingNew(false); }}
              className={`run-card ${selectedRun === run.id ? 'active' : ''}`}
              style={{
                padding: '12px 16px',
                borderRadius: '8px',
                marginBottom: '8px',
                cursor: 'pointer',
                background: selectedRun === run.id ? 'var(--accent-vibrant)' : 'rgba(255,255,255,0.02)',
                border: `1px solid ${selectedRun === run.id ? 'var(--accent-primary)' : 'rgba(255,255,255,0.05)'}`,
                position: 'relative',
              }}
              whileHover={{ x: 4 }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: '0.85rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{run.id}</span>
                <div style={{ display: 'flex', gap: '6px' }}>
                   {run.status === 'active' ? (
                    <Activity size={12} className="spin" color="var(--accent-lime)" />
                  ) : null}
                  <button onClick={(e) => handleDeleteRun(e, run.id)} style={{ background: 'none', border: 'none', color: 'var(--accent-pink)', opacity: 0.4 }}>
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>

              <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                <div className="run-tag" style={{ fontSize: '0.6rem' }}><Clock size={10} /> {new Date(run.created_at).toLocaleDateString()}</div>
                {run.eval_pck && (
                  <div className="run-tag" style={{ fontSize: '0.6rem', background: 'rgba(194, 239, 78, 0.1)', color: 'var(--accent-lime)' }}>PCK: {(run.eval_pck * 100).toFixed(1)}%</div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Run Detail Area (Right) */}
      <div className="run-detail-area" style={{ flex: 1, background: 'var(--bg-primary)', position: 'relative', overflow: 'hidden' }}>
        <AnimatePresence mode="wait">
          {isStartingNew ? (
            <motion.div 
              key="new-training"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              style={{ padding: '40px', height: '100%', overflowY: 'auto' }}
            >
              <TrainingForm onStarted={async () => { 
                setIsStartingNew(false); 
                await fetchRuns();
                // Immediately check status to get the new run_id
                const status = await getTrainingStatus();
                if (status.run_id) setSelectedRun(status.run_id);
              }} />
            </motion.div>
          ) : selectedRun ? (
            <motion.div 
              key={selectedRun}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{ padding: '40px', height: '100%', display: 'flex', flexDirection: 'column' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
                <div style={{ display: 'flex', gap: '24px' }}>
                  <button 
                    className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
                    onClick={() => setActiveTab('analysis')}
                  >
                    <Target size={18} />
                    Analysis
                  </button>
                  <button 
                    className={`tab-btn ${activeTab === 'inference' ? 'active' : ''}`}
                    onClick={() => setActiveTab('inference')}
                  >
                    <Eye size={18} />
                    Inference
                  </button>
                </div>
                <div className="run-id-badge" style={{ fontSize: '0.7rem' }}>{selectedRun}</div>
              </div>

              <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
                <AnimatePresence mode="wait">
                  {activeTab === 'analysis' ? (
                    <motion.div key="analysis" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <RunAnalysis 
                        key={selectedRun}
                        details={runDetails || { id: selectedRun }} 
                        isActive={trainingStatus?.is_running && trainingStatus.run_id === selectedRun}
                        trainingStatus={trainingStatus}
                      />
                    </motion.div>
                  ) : (
                    <motion.div key="inference" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ height: '100%' }}>
                      <RunInference runId={selectedRun} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="empty-state-hub"
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '20px', opacity: 0.5 }}
            >
              <Info size={64} />
              <div style={{ textAlign: 'center' }}>
                <h3 className="text-uppercase">Select an Experiment</h3>
                <p>Choose a run from the list or start a new training session.</p>
              </div>
              <button className="btn-lime" onClick={() => setIsStartingNew(true)}>
                <Plus size={18} /> Start New Run
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <style>{`
        .run-card.active {
          box-shadow: 0 4px 20px rgba(106, 95, 193, 0.3);
        }
        .run-tag {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: 0.7rem;
          padding: 2px 8px;
          background: rgba(255,255,255,0.05);
          border-radius: 4px;
          color: var(--text-secondary);
          font-weight: 600;
        }
        .tab-btn {
          background: none;
          border: none;
          color: var(--text-secondary);
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 1.1rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 1px;
          padding: 8px 0;
          border-bottom: 3px solid transparent;
          transition: all 0.2s;
          opacity: 0.6;
        }
        .tab-btn:hover { opacity: 1; }
        .tab-btn.active {
          color: var(--accent-lime);
          border-bottom-color: var(--accent-lime);
          opacity: 1;
        }
        .run-id-badge {
          padding: 4px 12px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-purple);
          border-radius: 20px;
          font-family: var(--font-mono);
          font-size: 0.8rem;
          color: var(--accent-primary);
        }
        .spin { animation: spin 2s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
};

export default Overview;
