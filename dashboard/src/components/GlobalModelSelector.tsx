import React, { useState, useEffect } from 'react';
import { Layers, RefreshCw, ChevronDown } from 'lucide-react';
import { useGlobalState } from '../context/GlobalStateContext';
import { getModels, getRuns } from '../services/api';

const GlobalModelSelector: React.FC = () => {
  const { selectedRun, selectedModel, setSelectedRun, setSelectedModel } = useGlobalState();
  const [runs, setRuns] = useState<{id: string}[]>([]);
  const [models, setModels] = useState<{name: string}[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [runsData, modelsData] = await Promise.all([
          getRuns(),
          getModels()
        ]);
        setRuns(runsData.runs || []);
        setModels(modelsData.models || []);
      } catch (err) {
        console.error('Failed to load models/runs for selector:', err);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="custom-dropdown-container" style={{ position: 'relative', width: '250px' }}>
      <div 
        className={`glass custom-dropdown-trigger ${isDropdownOpen ? 'open' : ''}`}
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        style={{ 
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          minHeight: '36px',
          borderRadius: '8px',
          border: '1px solid var(--border-purple)',
          background: 'rgba(255,255,255,0.05)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
          {selectedRun ? (
            <>
              <RefreshCw size={14} className="text-secondary" />
              <span style={{ fontSize: '0.85rem', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>Run: {selectedRun}</span>
            </>
          ) : selectedModel ? (
            <>
              <Layers size={14} className="text-secondary" />
              <span style={{ fontSize: '0.85rem', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>{selectedModel}</span>
            </>
          ) : (
            <span className="text-secondary" style={{ fontSize: '0.85rem' }}>Select Active Model...</span>
          )}
        </div>
        <ChevronDown size={14} style={{ transform: isDropdownOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s', marginLeft: '8px' }} />
      </div>

      {isDropdownOpen && (
        <div className="glass dropdown-menu" style={{ 
          position: 'absolute',
          top: 'calc(100% + 4px)',
          left: 0,
          right: 0,
          zIndex: 1000,
          borderRadius: '8px',
          padding: '4px',
          maxHeight: '400px',
          overflowY: 'auto',
          boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
          border: '1px solid var(--border-purple)',
          background: 'var(--bg-secondary)'
        }}>
          <div 
            className={`dropdown-item ${!selectedRun && !selectedModel ? 'active' : ''}`}
            onClick={() => {
              setSelectedRun('');
              setSelectedModel('');
              setIsDropdownOpen(false);
            }}
          >
            <span style={{ marginLeft: '24px' }}>None</span>
          </div>

          {runs.length > 0 && (
            <>
              <div className="micro-label text-uppercase" style={{ padding: '8px', opacity: 0.5, fontSize: '0.6rem' }}>Training Runs</div>
              {runs.map(r => (
                <div 
                  key={r.id}
                  className={`dropdown-item ${selectedRun === r.id ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedRun(r.id);
                    setSelectedModel('');
                    setIsDropdownOpen(false);
                  }}
                >
                  <RefreshCw size={14} />
                  {r.id}
                </div>
              ))}
            </>
          )}
          
          {models.length > 0 && (
            <>
              <div className="micro-label text-uppercase" style={{ padding: '8px', opacity: 0.5, fontSize: '0.6rem' }}>Global Models</div>
              {models.map(m => (
                <div 
                  key={m.name}
                  className={`dropdown-item ${selectedModel === m.name ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedModel(m.name);
                    setSelectedRun('');
                    setIsDropdownOpen(false);
                  }}
                >
                  <Layers size={14} />
                  {m.name}
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default GlobalModelSelector;
