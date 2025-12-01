const { useState, useEffect } = React;

function App() {
    const [activeTab, setActiveTab] = useState('training');
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            setModels(data);
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    };

    return (
        <div className="app-wrapper">
            <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
                <div className="sidebar-header">
                    <h2>Nowcasting</h2>
                    <button 
                        className="sidebar-toggle"
                        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                    >
                        {sidebarCollapsed ? '☰' : '✕'}
                    </button>
                </div>
                <ul className="sidebar-menu">
                    <li className={`sidebar-menu-item ${activeTab === 'training' ? 'active' : ''}`}>
                        <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('training'); }}>
                            Training Dashboard
                        </a>
                    </li>
                    <li className={`sidebar-menu-item ${activeTab === 'inference' ? 'active' : ''}`}>
                        <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('inference'); }}>
                            Inference & Reports
                        </a>
                    </li>
                </ul>
            </div>
            <div className={`main-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
                {activeTab === 'training' && (
                    <TrainingTab models={models} onModelTrained={loadModels} />
                )}
                {activeTab === 'inference' && (
                    <InferenceTab models={models} />
                )}
            </div>
        </div>
    );
}

function TrainingTab({ models, onModelTrained }) {
    const [file, setFile] = useState(null);
    const [experimentId, setExperimentId] = useState('default');
    const [modelType, setModelType] = useState('dfm');
    const [modelName, setModelName] = useState('');
    const [training, setTraining] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('');
    const [error, setError] = useState(null);
    const [jobId, setJobId] = useState(null);
    const [ws, setWs] = useState(null);
    const [experiments, setExperiments] = useState([]);
    const [configTab, setConfigTab] = useState('general');
    const [showCreateExperiment, setShowCreateExperiment] = useState(false);
    const [newExperimentId, setNewExperimentId] = useState('');
    const [newExperimentType, setNewExperimentType] = useState('dfm');
    
    // Config states
    const [generalConfig, setGeneralConfig] = useState('');
    const [generalConfigData, setGeneralConfigData] = useState({});
    const [seriesConfig, setSeriesConfig] = useState('');
    const [blockConfig, setBlockConfig] = useState('');
    const [seriesConfigName, setSeriesConfigName] = useState('default');
    const [blockConfigName, setBlockConfigName] = useState('default');
    const [seriesConfigs, setSeriesConfigs] = useState([]);
    const [blockConfigs, setBlockConfigs] = useState([]);
    
    // Dashboard states
    const [searchQuery, setSearchQuery] = useState('');
    const [refreshing, setRefreshing] = useState(false);
    const [dashboardStats, setDashboardStats] = useState(null);
    const [trainingJobs, setTrainingJobs] = useState([]);
    const [experimentUsage, setExperimentUsage] = useState({});
    
    // Helper function to parse YAML-like config into object
    const parseConfigYAML = (yamlText) => {
        const data = {};
        const lines = yamlText.split('\n');
        let inDefaults = false;
        let defaults = [];
        
        for (let line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('#') || !trimmed) continue;
            
            if (trimmed === 'defaults:') {
                inDefaults = true;
                continue;
            }
            
            if (inDefaults && trimmed.startsWith('-')) {
                defaults.push(trimmed.substring(1).trim());
                continue;
            }
            
            if (inDefaults && trimmed && !trimmed.startsWith('-')) {
                inDefaults = false;
            }
            
            if (!inDefaults && trimmed.includes(':')) {
                const colonIndex = trimmed.indexOf(':');
                const key = trimmed.substring(0, colonIndex).trim();
                let value = trimmed.substring(colonIndex + 1).trim();
                
                // Remove quotes
                if ((value.startsWith('"') && value.endsWith('"')) || 
                    (value.startsWith("'") && value.endsWith("'"))) {
                    value = value.slice(1, -1);
                }
                
                // Parse boolean
                if (value === 'true') value = true;
                else if (value === 'false') value = false;
                // Parse number
                else if (!isNaN(value) && value !== '') {
                    if (value.includes('.')) value = parseFloat(value);
                    else value = parseInt(value);
                }
                // Parse array
                else if (value.startsWith('[') && value.endsWith(']')) {
                    try {
                        value = JSON.parse(value);
                    } catch (e) {
                        // Keep as string if parsing fails
                    }
                }
                
                data[key] = value;
            }
        }
        
        return { data, defaults };
    };
    
    // Helper function to convert object back to YAML
    const objectToYAML = (data, defaults = []) => {
        let yaml = '';
        
        if (defaults.length > 0) {
            yaml += 'defaults:\n';
            defaults.forEach(d => {
                yaml += `  - ${d}\n`;
            });
            yaml += '\n';
        }
        
        // Group by sections
        const estimationParams = ['ar_lag', 'threshold', 'max_iter', 'nan_method', 'nan_k', 'clock'];
        const numericalStability = ['clip_ar_coefficients', 'ar_clip_min', 'ar_clip_max', 'warn_on_ar_clip',
            'clip_data_values', 'data_clip_threshold', 'warn_on_data_clip',
            'use_regularization', 'regularization_scale', 'min_eigenvalue', 'max_eigenvalue', 'warn_on_regularization',
            'use_damped_updates', 'damping_factor', 'warn_on_damped_update'];
        const ddfmParams = Object.keys(data).filter(k => k.startsWith('ddfm_'));
        
        // Estimation parameters
        const hasEstimation = estimationParams.some(k => data.hasOwnProperty(k));
        if (hasEstimation) {
            yaml += '# Estimation Parameters\n';
            estimationParams.forEach(k => {
                if (data.hasOwnProperty(k)) {
                    const val = data[k];
                    if (typeof val === 'string' && !val.match(/^[0-9.e+-]+$/)) {
                        yaml += `${k}: '${val}'\n`;
                    } else {
                        yaml += `${k}: ${val}\n`;
                    }
                }
            });
            yaml += '\n';
        }
        
        // Numerical stability
        const hasStability = numericalStability.some(k => data.hasOwnProperty(k));
        if (hasStability) {
            yaml += '# Numerical Stability Parameters\n';
            numericalStability.forEach(k => {
                if (data.hasOwnProperty(k)) {
                    const val = data[k];
                    if (typeof val === 'string' && !val.match(/^[0-9.e+-]+$/)) {
                        yaml += `${k}: '${val}'\n`;
                    } else {
                        yaml += `${k}: ${val}\n`;
                    }
                }
            });
            yaml += '\n';
        }
        
        // DDFM parameters
        if (ddfmParams.length > 0) {
            yaml += '# DDFM-specific parameters\n';
            ddfmParams.forEach(k => {
                const val = data[k];
                if (Array.isArray(val)) {
                    yaml += `${k}: [${val.join(', ')}]\n`;
                } else if (typeof val === 'string' && !val.match(/^[0-9.e+-]+$/)) {
                    yaml += `${k}: '${val}'\n`;
                } else {
                    yaml += `${k}: ${val}\n`;
                }
            });
            yaml += '\n';
        }
        
        // Other parameters
        const otherParams = Object.keys(data).filter(k => 
            !estimationParams.includes(k) && 
            !numericalStability.includes(k) && 
            !k.startsWith('ddfm_')
        );
        if (otherParams.length > 0) {
            otherParams.forEach(k => {
                const val = data[k];
                if (Array.isArray(val)) {
                    yaml += `${k}: [${val.join(', ')}]\n`;
                } else if (typeof val === 'string' && !val.match(/^[0-9.e+-]+$/)) {
                    yaml += `${k}: '${val}'\n`;
                } else {
                    yaml += `${k}: ${val}\n`;
                }
            });
        }
        
        return yaml;
    };

    useEffect(() => {
        loadExperiments();
        loadSeriesConfigs();
        loadBlockConfigs();
    }, []);

    useEffect(() => {
        if (experimentId) {
            loadExperiment(experimentId);
        }
    }, [experimentId]);

    const loadExperiments = async () => {
        try {
            const response = await fetch('/api/experiments');
            const data = await response.json();
            setExperiments(data);
            if (data.length > 0) {
                const defaultExp = data.find(e => e.experiment_id === 'default') || data[0];
                setExperimentId(defaultExp.experiment_id);
                setModelType(defaultExp.model_type);
            }
        } catch (error) {
            console.error('Failed to load experiments:', error);
        }
    };

    const loadExperiment = async (expId) => {
        try {
            const response = await fetch(`/api/experiment/${expId}`);
            const data = await response.json();
            setModelType(data.model_type);
            
            // Parse experiment config to extract series and block references
            const expConfig = data.content;
            const lines = expConfig.split('\n');
            let seriesRef = 'default';
            let blockRef = 'default';
            let inDefaults = false;
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line.startsWith('defaults:')) {
                    inDefaults = true;
                    continue;
                }
                if (inDefaults && line.startsWith('-')) {
                    if (line.includes('series:')) {
                        const match = line.match(/series:\s*(\w+)/);
                        if (match) seriesRef = match[1];
                    }
                    if (line.includes('blocks:')) {
                        const match = line.match(/blocks:\s*(\w+)/);
                        if (match) blockRef = match[1];
                    }
                }
                if (line && !line.startsWith('#') && !line.startsWith('-') && !line.startsWith('defaults:')) {
                    inDefaults = false;
                }
            }
            
            setSeriesConfigName(seriesRef);
            setBlockConfigName(blockRef);
            
            // Load referenced configs
            if (seriesRef) {
                try {
                    const seriesResp = await fetch(`/api/series-config/${seriesRef}`);
                    const seriesData = await seriesResp.json();
                    setSeriesConfig(seriesData.content);
                } catch (e) {
                    console.error('Failed to load series config:', e);
                }
            }
            
            if (data.model_type === 'dfm' && blockRef) {
                try {
                    const blockResp = await fetch(`/api/block-config/${blockRef}`);
                    const blockData = await blockResp.json();
                    setBlockConfig(blockData.content);
                } catch (e) {
                    console.error('Failed to load block config:', e);
                }
            }
            
            // Extract general config (experiment config itself)
            setGeneralConfig(expConfig);
            // Parse config into structured data
            const parsed = parseConfigYAML(expConfig);
            setGeneralConfigData(parsed.data);
        } catch (error) {
            console.error('Failed to load experiment:', error);
        }
    };
    
    const updateConfigField = (key, value) => {
        setGeneralConfigData(prev => ({
            ...prev,
            [key]: value
        }));
    };

    const loadSeriesConfigs = async () => {
        try {
            const response = await fetch('/api/series-configs');
            const data = await response.json();
            setSeriesConfigs(data);
        } catch (error) {
            console.error('Failed to load series configs:', error);
        }
    };

    const loadBlockConfigs = async () => {
        try {
            const response = await fetch('/api/block-configs');
            const data = await response.json();
            setBlockConfigs(data);
        } catch (error) {
            console.error('Failed to load block configs:', error);
        }
    };

    const loadSeriesConfig = async (name) => {
        try {
            const response = await fetch(`/api/series-config/${name}`);
            const data = await response.json();
            setSeriesConfig(data.content);
        } catch (error) {
            console.error('Failed to load series config:', error);
        }
    };

    const loadBlockConfig = async (name) => {
        try {
            const response = await fetch(`/api/block-config/${name}`);
            const data = await response.json();
            setBlockConfig(data.content);
        } catch (error) {
            console.error('Failed to load block config:', error);
        }
    };

    const handleFileUpload = async (e) => {
        const selectedFile = e.target.files[0];
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            setFile(data.path);
        } catch (error) {
            setError('File upload failed: ' + error.message);
        }
    };

    const createExperiment = async () => {
        if (!newExperimentId) {
            setError('Please enter an experiment ID');
            return;
        }
        
        // Create default experiment config based on model type
        let defaultConfig = '';
        if (newExperimentType === 'dfm') {
            defaultConfig = `defaults:
  - /default
  - series: default
  - blocks: default
  - _self_

# Override estimation parameters if needed
# threshold: 1e-5
# max_iter: 5000
`;
        } else {
            defaultConfig = `defaults:
  - /default
  - series: default
  - _self_

# DDFM-specific parameters (Placeholder)
ddfm_encoder_layers: [64, 32]
ddfm_num_factors: 1
ddfm_activation: 'tanh'
ddfm_use_batch_norm: true
ddfm_learning_rate: 0.001
ddfm_epochs: 100
ddfm_batch_size: 32
ddfm_factor_order: 1
ddfm_use_idiosyncratic: true
ddfm_min_obs_idio: 5
`;
        }
        
        try {
            const response = await fetch(`/api/experiment/${newExperimentId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_type: newExperimentType,
                    content: defaultConfig
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to create experiment');
            }
            
            setShowCreateExperiment(false);
            setNewExperimentId('');
            loadExperiments();
            setExperimentId(newExperimentId);
        } catch (error) {
            setError('Failed to create experiment: ' + error.message);
        }
    };

    const saveGeneralConfig = async () => {
        try {
            // Get defaults from current config
            const parsed = parseConfigYAML(generalConfig);
            const yamlContent = objectToYAML(generalConfigData, parsed.defaults);
            
            const response = await fetch(`/api/experiment/${experimentId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_type: modelType,
                    content: yamlContent
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save config');
            }
            
            alert('General config saved successfully');
            loadExperiment(experimentId);
        } catch (error) {
            setError('Failed to save general config: ' + error.message);
        }
    };

    const saveSeriesConfig = async () => {
        try {
            const response = await fetch(`/api/series-config/${seriesConfigName}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: seriesConfig })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save series config');
            }
            
            alert('Series config saved successfully');
        } catch (error) {
            setError('Failed to save series config: ' + error.message);
        }
    };

    const saveBlockConfig = async () => {
        try {
            const response = await fetch(`/api/block-config/${blockConfigName}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: blockConfig })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to save block config');
            }
            
            alert('Block config saved successfully');
        } catch (error) {
            setError('Failed to save block config: ' + error.message);
        }
    };

    const startTraining = async () => {
        if (!file) {
            setError('Please upload a data file');
            return;
        }

        setTraining(true);
        setError(null);
        setProgress(0);
        setStatus('Starting training...');

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: modelName || undefined,
                    model_type: modelType,
                    experiment_id: experimentId,
                    data_path: file
                })
            });

            const data = await response.json();
            setJobId(data.job_id);

            const wsUrl = `ws://${window.location.host}/ws/train/${data.job_id}`;
            const websocket = new WebSocket(wsUrl);

            websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'progress') {
                    setProgress(message.data.progress);
                    setStatus(message.data.message);
                } else if (message.type === 'complete') {
                    setTraining(false);
                    setStatus('Training completed!');
                    if (message.data.status === 'failed') {
                        setError(message.data.error);
                    } else {
                        onModelTrained();
                    }
                    // Refresh dashboard stats when training completes
                    loadDashboardStats();
                    loadTrainingJobs();
                    loadExperimentUsage();
                    websocket.close();
                }
            };

            websocket.onerror = (error) => {
                setError('WebSocket error');
                setTraining(false);
            };

            setWs(websocket);
        } catch (error) {
            setError('Training failed: ' + error.message);
            setTraining(false);
        }
    };

    const loadDashboardStats = async () => {
        try {
            const response = await fetch('/api/dashboard/stats');
            const data = await response.json();
            setDashboardStats(data);
        } catch (error) {
            console.error('Failed to load dashboard stats:', error);
        }
    };

    const loadTrainingJobs = async () => {
        try {
            const response = await fetch('/api/dashboard/training-jobs');
            const data = await response.json();
            setTrainingJobs(data);
        } catch (error) {
            console.error('Failed to load training jobs:', error);
        }
    };

    const loadExperimentUsage = async () => {
        try {
            const response = await fetch('/api/dashboard/experiment-usage');
            const data = await response.json();
            setExperimentUsage(data);
        } catch (error) {
            console.error('Failed to load experiment usage:', error);
        }
    };

    useEffect(() => {
        loadDashboardStats();
        loadTrainingJobs();
        loadExperimentUsage();
        
        // Auto-refresh every 30 seconds
        const interval = setInterval(() => {
            loadDashboardStats();
            loadTrainingJobs();
            loadExperimentUsage();
        }, 30000);
        
        return () => clearInterval(interval);
    }, []);

    const handleRefresh = async () => {
        setRefreshing(true);
        await loadExperiments();
        await onModelTrained();
        await loadDashboardStats();
        await loadTrainingJobs();
        await loadExperimentUsage();
        setRefreshing(false);
    };

    return (
        <div>
            <div className="dashboard-header">
                <div className="dashboard-header-left">
                    <h1 className="dashboard-title">Training Dashboard</h1>
                    <input
                        type="text"
                        className="dashboard-search"
                        placeholder="Filter models, experiments..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>
                <div className="dashboard-header-right">
                    <button className="dashboard-btn secondary" onClick={handleRefresh} disabled={refreshing}>
                        {refreshing ? 'Refreshing...' : '↻ Refresh'}
                    </button>
                </div>
            </div>
            <div className="dashboard-grid">
                {dashboardStats && (
                    <>
                        <div className="kpi-card">
                            <div className="kpi-value">{dashboardStats.total_models}</div>
                            <div className="kpi-label">Total Models</div>
                        </div>
                        <div className={`kpi-card ${dashboardStats.active_training_jobs > 0 ? 'warning' : 'success'}`}>
                            <div className="kpi-value">{dashboardStats.active_training_jobs}</div>
                            <div className="kpi-label">Active Training</div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-value">{dashboardStats.total_experiments}</div>
                            <div className="kpi-label">Experiments</div>
                        </div>
                        <div className="kpi-card success">
                            <div className="kpi-value">{dashboardStats.recent_success_rate.toFixed(1)}%</div>
                            <div className="kpi-label">Success Rate</div>
                            <div className="kpi-progress">
                                <div className="kpi-progress-fill" style={{width: `${dashboardStats.recent_success_rate}%`}}></div>
                            </div>
                        </div>
                    </>
                )}
                <div className="dashboard-panel">
                    <div className="panel-header">
                        <h3 className="panel-title">Upload Data</h3>
                    </div>
                    <div className="panel-content">
                        <div className="form-group">
                            <label>CSV Data File</label>
                            <input type="file" accept=".csv" onChange={handleFileUpload} />
                            {file && <p className="success">File uploaded: {file}</p>}
                        </div>
                    </div>
                </div>

                <div className="dashboard-panel" style={{gridColumn: 'span 2'}}>
                    <div className="panel-header">
                        <h3 className="panel-title">Recent Training Jobs</h3>
                    </div>
                    <div className="panel-content">
                        <img 
                            src="/api/dashboard/chart/training-timeline" 
                            alt="Training Timeline"
                            style={{width: '100%', height: 'auto'}}
                            onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.nextSibling.style.display = 'block';
                            }}
                        />
                        <div style={{display: 'none', color: '#718096', textAlign: 'center', padding: '20px'}}>
                            Loading chart...
                        </div>
                    </div>
                </div>

                <div className="dashboard-panel">
                    <div className="panel-header">
                        <h3 className="panel-title">Models by Type</h3>
                    </div>
                    <div className="panel-content">
                        <img 
                            src="/api/dashboard/chart/model-distribution" 
                            alt="Model Distribution"
                            style={{width: '100%', height: 'auto'}}
                            onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.nextSibling.style.display = 'block';
                            }}
                        />
                        <div style={{display: 'none', color: '#718096', textAlign: 'center', padding: '20px'}}>
                            Loading chart...
                        </div>
                    </div>
                </div>

                <div className="dashboard-panel">
                    <div className="panel-header">
                        <h3 className="panel-title">Most Used Experiments</h3>
                    </div>
                    <div className="panel-content">
                        <img 
                            src="/api/dashboard/chart/experiment-usage" 
                            alt="Experiment Usage"
                            style={{width: '100%', height: 'auto'}}
                            onError={(e) => {
                                e.target.style.display = 'none';
                                e.target.nextSibling.style.display = 'block';
                            }}
                        />
                        <div style={{display: 'none', color: '#718096', textAlign: 'center', padding: '20px'}}>
                            Loading chart...
                        </div>
                    </div>
                </div>

                <div className="dashboard-panel" style={{gridColumn: 'span 2'}}>
                    <div className="panel-header">
                        <h3 className="panel-title">Configuration</h3>
                    </div>
                    <div className="panel-content">
                        <div className="form-group">
                            <label>Experiment</label>
                            <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
                                <select
                                    value={experimentId}
                                    onChange={(e) => setExperimentId(e.target.value)}
                                    style={{flex: 1}}
                                >
                                    {experiments.map(exp => (
                                        <option key={exp.experiment_id} value={exp.experiment_id}>
                                            {exp.experiment_id} ({exp.model_type.toUpperCase()})
                                        </option>
                                    ))}
                                </select>
                                <button onClick={() => setShowCreateExperiment(true)}>Create New</button>
                            </div>
                        </div>
                        <div className="form-group">
                            <label>Model Type</label>
                            <select value={modelType} disabled>
                                <option value="dfm">DFM (Dynamic Factor Model)</option>
                                <option value="ddfm">DDFM (Deep Dynamic Factor Model)</option>
                            </select>
                            <p style={{color: '#666', fontSize: '12px', marginTop: '5px'}}>
                                Model type is determined by the selected experiment
                            </p>
                        </div>

                        {showCreateExperiment && (
                            <div style={{border: '2px solid #007bff', padding: '15px', borderRadius: '4px', marginTop: '15px', marginBottom: '15px'}}>
                                <h3>Create New Experiment</h3>
                                <div className="form-group">
                                    <label>Experiment ID</label>
                                    <input
                                        type="text"
                                        value={newExperimentId}
                                        onChange={(e) => setNewExperimentId(e.target.value)}
                                        placeholder="e.g., exp4, my_experiment"
                                    />
                                </div>
                                <div className="form-group">
                                    <label>Model Type</label>
                                    <select value={newExperimentType} onChange={(e) => setNewExperimentType(e.target.value)}>
                                        <option value="dfm">DFM (Dynamic Factor Model)</option>
                                        <option value="ddfm">DDFM (Deep Dynamic Factor Model)</option>
                                    </select>
                                </div>
                                <div style={{display: 'flex', gap: '10px'}}>
                                    <button onClick={createExperiment}>Create</button>
                                    <button onClick={() => {setShowCreateExperiment(false); setNewExperimentId('');}}>Cancel</button>
                                </div>
                            </div>
                        )}

                        <div style={{display: 'flex', gap: '10px', marginBottom: '20px', marginTop: '20px', borderTop: '1px solid #ddd', borderBottom: '1px solid #ddd', paddingTop: '10px', paddingBottom: '10px'}}>
                            <button
                                className={`tab ${configTab === 'general' ? 'active' : ''}`}
                                onClick={() => setConfigTab('general')}
                            >
                                General Config
                            </button>
                            <button
                                className={`tab ${configTab === 'series' ? 'active' : ''}`}
                                onClick={() => setConfigTab('series')}
                            >
                                Series Config
                            </button>
                            {modelType === 'dfm' && (
                                <button
                                    className={`tab ${configTab === 'blocks' ? 'active' : ''}`}
                                    onClick={() => setConfigTab('blocks')}
                                >
                                    Block Config
                                </button>
                            )}
                        </div>

                        {configTab === 'general' && (
                            <div>
                                <div className="config-section">
                                    <div className="config-section-header">
                                        <h3>Estimation Parameters</h3>
                                    </div>
                                    <table className="config-table">
                                        <thead>
                                            <tr>
                                                <th>Parameter</th>
                                                <th>Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">AR Lag</div>
                                                    <div className="config-param-help">AR lag for factor transition equation (typically 1)</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        className="config-param-control"
                                                        value={generalConfigData.ar_lag || 1}
                                                        onChange={(e) => updateConfigField('ar_lag', parseInt(e.target.value) || 1)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Convergence Threshold</div>
                                                    <div className="config-param-help">EM convergence threshold (e.g., 1e-5)</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="text"
                                                        className="config-param-control"
                                                        value={generalConfigData.threshold || '1e-5'}
                                                        onChange={(e) => updateConfigField('threshold', e.target.value)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Max Iterations</div>
                                                    <div className="config-param-help">Maximum EM iterations</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        className="config-param-control"
                                                        value={generalConfigData.max_iter || 5000}
                                                        onChange={(e) => updateConfigField('max_iter', parseInt(e.target.value) || 5000)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">NaN Method</div>
                                                    <div className="config-param-help">Method for handling NaN values</div>
                                                </td>
                                                <td>
                                                    <select
                                                        className="config-param-control"
                                                        value={generalConfigData.nan_method || 2}
                                                        onChange={(e) => updateConfigField('nan_method', parseInt(e.target.value))}
                                                    >
                                                        <option value={1}>1</option>
                                                        <option value={2}>2 (Spline interpolation)</option>
                                                        <option value={3}>3</option>
                                                        <option value={4}>4</option>
                                                        <option value={5}>5</option>
                                                    </select>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">NaN K</div>
                                                    <div className="config-param-help">Spline parameter for interpolation</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        className="config-param-control"
                                                        value={generalConfigData.nan_k || 3}
                                                        onChange={(e) => updateConfigField('nan_k', parseInt(e.target.value) || 3)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Clock Frequency</div>
                                                    <div className="config-param-help">Base frequency for all latent factors</div>
                                                </td>
                                                <td>
                                                    <select
                                                        className="config-param-control"
                                                        value={generalConfigData.clock || 'm'}
                                                        onChange={(e) => updateConfigField('clock', e.target.value)}
                                                    >
                                                        <option value="d">Daily (d)</option>
                                                        <option value="w">Weekly (w)</option>
                                                        <option value="m">Monthly (m)</option>
                                                        <option value="q">Quarterly (q)</option>
                                                        <option value="sa">Semi-annual (sa)</option>
                                                        <option value="a">Annual (a)</option>
                                                    </select>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                <div className="config-section">
                                    <div className="config-section-header">
                                        <h3>AR Coefficient Clipping</h3>
                                    </div>
                                    <table className="config-table">
                                        <thead>
                                            <tr>
                                                <th>Parameter</th>
                                                <th>Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Enable AR Clipping</div>
                                                    <div className="config-param-help">Ensures stationarity</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.clip_ar_coefficients !== false}
                                                            onChange={(e) => updateConfigField('clip_ar_coefficients', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.clip_ar_coefficients !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.clip_ar_coefficients !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">AR Clip Min</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        step="0.01"
                                                        className="config-param-control"
                                                        value={generalConfigData.ar_clip_min || -0.99}
                                                        onChange={(e) => updateConfigField('ar_clip_min', parseFloat(e.target.value) || -0.99)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">AR Clip Max</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        step="0.01"
                                                        className="config-param-control"
                                                        value={generalConfigData.ar_clip_max || 0.99}
                                                        onChange={(e) => updateConfigField('ar_clip_max', parseFloat(e.target.value) || 0.99)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Warn on AR Clip</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.warn_on_ar_clip !== false}
                                                            onChange={(e) => updateConfigField('warn_on_ar_clip', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.warn_on_ar_clip !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.warn_on_ar_clip !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                <div className="config-section">
                                    <div className="config-section-header">
                                        <h3>Data Value Clipping</h3>
                                    </div>
                                    <table className="config-table">
                                        <thead>
                                            <tr>
                                                <th>Parameter</th>
                                                <th>Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Enable Data Clipping</div>
                                                    <div className="config-param-help">Handles extreme outliers</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.clip_data_values !== false}
                                                            onChange={(e) => updateConfigField('clip_data_values', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.clip_data_values !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.clip_data_values !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Data Clip Threshold</div>
                                                    <div className="config-param-help">Clip values beyond this many standard deviations</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        step="0.1"
                                                        className="config-param-control"
                                                        value={generalConfigData.data_clip_threshold || 100.0}
                                                        onChange={(e) => updateConfigField('data_clip_threshold', parseFloat(e.target.value) || 100.0)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Warn on Data Clip</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.warn_on_data_clip !== false}
                                                            onChange={(e) => updateConfigField('warn_on_data_clip', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.warn_on_data_clip !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.warn_on_data_clip !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                <div className="config-section">
                                    <div className="config-section-header">
                                        <h3>Regularization</h3>
                                    </div>
                                    <table className="config-table">
                                        <thead>
                                            <tr>
                                                <th>Parameter</th>
                                                <th>Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Use Regularization</div>
                                                    <div className="config-param-help">Prevents ill-conditioned matrices</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.use_regularization !== false}
                                                            onChange={(e) => updateConfigField('use_regularization', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.use_regularization !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.use_regularization !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Regularization Scale</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="text"
                                                        className="config-param-control"
                                                        value={generalConfigData.regularization_scale || '1e-5'}
                                                        onChange={(e) => updateConfigField('regularization_scale', e.target.value)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Min Eigenvalue</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="text"
                                                        className="config-param-control"
                                                        value={generalConfigData.min_eigenvalue || '1e-8'}
                                                        onChange={(e) => updateConfigField('min_eigenvalue', e.target.value)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Max Eigenvalue</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="text"
                                                        className="config-param-control"
                                                        value={generalConfigData.max_eigenvalue || '1e6'}
                                                        onChange={(e) => updateConfigField('max_eigenvalue', e.target.value)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Warn on Regularization</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.warn_on_regularization !== false}
                                                            onChange={(e) => updateConfigField('warn_on_regularization', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.warn_on_regularization !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.warn_on_regularization !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                <div className="config-section">
                                    <div className="config-section-header">
                                        <h3>Damped Updates</h3>
                                    </div>
                                    <table className="config-table">
                                        <thead>
                                            <tr>
                                                <th>Parameter</th>
                                                <th>Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Use Damped Updates</div>
                                                    <div className="config-param-help">Prevents likelihood decreases</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.use_damped_updates !== false}
                                                            onChange={(e) => updateConfigField('use_damped_updates', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.use_damped_updates !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.use_damped_updates !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Damping Factor</div>
                                                    <div className="config-param-help">80% new, 20% old when likelihood decreases</div>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        step="0.1"
                                                        className="config-param-control"
                                                        value={generalConfigData.damping_factor || 0.8}
                                                        onChange={(e) => updateConfigField('damping_factor', parseFloat(e.target.value) || 0.8)}
                                                    />
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <div className="config-param-name">Warn on Damped Update</div>
                                                </td>
                                                <td>
                                                    <div className="config-param-group">
                                                        <input
                                                            type="checkbox"
                                                            className="config-param-control"
                                                            checked={generalConfigData.warn_on_damped_update !== false}
                                                            onChange={(e) => updateConfigField('warn_on_damped_update', e.target.checked)}
                                                        />
                                                        <span className={`config-badge ${generalConfigData.warn_on_damped_update !== false ? 'enabled' : 'disabled'}`}>
                                                            {generalConfigData.warn_on_damped_update !== false ? 'Enabled' : 'Disabled'}
                                                        </span>
                                                    </div>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                {modelType === 'ddfm' && (
                                    <div className="config-section">
                                        <div className="config-section-header">
                                            <h3>DDFM-specific Parameters (Placeholder)</h3>
                                        </div>
                                        <table className="config-table">
                                            <thead>
                                                <tr>
                                                    <th>Parameter</th>
                                                    <th>Value</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Encoder Layers</div>
                                                        <div className="config-param-help">Hidden layer dimensions (comma-separated, e.g., 64, 32)</div>
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="text"
                                                            className="config-param-control"
                                                            value={Array.isArray(generalConfigData.ddfm_encoder_layers) ? generalConfigData.ddfm_encoder_layers.join(', ') : (generalConfigData.ddfm_encoder_layers || '64, 32')}
                                                            onChange={(e) => {
                                                                const layers = e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                                                                updateConfigField('ddfm_encoder_layers', layers);
                                                            }}
                                                        />
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Number of Factors</div>
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="number"
                                                            className="config-param-control"
                                                            value={generalConfigData.ddfm_num_factors || 1}
                                                            onChange={(e) => updateConfigField('ddfm_num_factors', parseInt(e.target.value) || 1)}
                                                        />
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Activation Function</div>
                                                    </td>
                                                    <td>
                                                        <select
                                                            className="config-param-control"
                                                            value={generalConfigData.ddfm_activation || 'tanh'}
                                                            onChange={(e) => updateConfigField('ddfm_activation', e.target.value)}
                                                        >
                                                            <option value="tanh">tanh</option>
                                                            <option value="relu">relu</option>
                                                            <option value="sigmoid">sigmoid</option>
                                                        </select>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Use Batch Normalization</div>
                                                    </td>
                                                    <td>
                                                        <div className="config-param-group">
                                                            <input
                                                                type="checkbox"
                                                                className="config-param-control"
                                                                checked={generalConfigData.ddfm_use_batch_norm !== false}
                                                                onChange={(e) => updateConfigField('ddfm_use_batch_norm', e.target.checked)}
                                                            />
                                                            <span className={`config-badge ${generalConfigData.ddfm_use_batch_norm !== false ? 'enabled' : 'disabled'}`}>
                                                                {generalConfigData.ddfm_use_batch_norm !== false ? 'Enabled' : 'Disabled'}
                                                            </span>
                                                        </div>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Learning Rate</div>
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="text"
                                                            className="config-param-control"
                                                            value={generalConfigData.ddfm_learning_rate || '0.001'}
                                                            onChange={(e) => updateConfigField('ddfm_learning_rate', e.target.value)}
                                                        />
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Epochs</div>
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="number"
                                                            className="config-param-control"
                                                            value={generalConfigData.ddfm_epochs || 100}
                                                            onChange={(e) => updateConfigField('ddfm_epochs', parseInt(e.target.value) || 100)}
                                                        />
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Batch Size</div>
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="number"
                                                            className="config-param-control"
                                                            value={generalConfigData.ddfm_batch_size || 32}
                                                            onChange={(e) => updateConfigField('ddfm_batch_size', parseInt(e.target.value) || 32)}
                                                        />
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Factor Order</div>
                                                        <div className="config-param-help">VAR lag order for factor dynamics</div>
                                                    </td>
                                                    <td>
                                                        <select
                                                            className="config-param-control"
                                                            value={generalConfigData.ddfm_factor_order || 2}
                                                            onChange={(e) => updateConfigField('ddfm_factor_order', parseInt(e.target.value))}
                                                        >
                                                            <option value={1}>1</option>
                                                            <option value={2}>2</option>
                                                        </select>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Use Idiosyncratic</div>
                                                        <div className="config-param-help">Model idiosyncratic components with AR(1) dynamics</div>
                                                    </td>
                                                    <td>
                                                        <div className="config-param-group">
                                                            <input
                                                                type="checkbox"
                                                                className="config-param-control"
                                                                checked={generalConfigData.ddfm_use_idiosyncratic !== false}
                                                                onChange={(e) => updateConfigField('ddfm_use_idiosyncratic', e.target.checked)}
                                                            />
                                                            <span className={`config-badge ${generalConfigData.ddfm_use_idiosyncratic !== false ? 'enabled' : 'disabled'}`}>
                                                                {generalConfigData.ddfm_use_idiosyncratic !== false ? 'Enabled' : 'Disabled'}
                                                            </span>
                                                        </div>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td>
                                                        <div className="config-param-name">Min Obs Idio</div>
                                                        <div className="config-param-help">Minimum observations for idio AR(1) estimation</div>
                                                    </td>
                                                    <td>
                                                        <input
                                                            type="number"
                                                            className="config-param-control"
                                                            value={generalConfigData.ddfm_min_obs_idio || 5}
                                                            onChange={(e) => updateConfigField('ddfm_min_obs_idio', parseInt(e.target.value) || 5)}
                                                        />
                                                    </td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                )}

                                <div style={{marginTop: '20px'}}>
                                    <button onClick={saveGeneralConfig}>Save General Config</button>
                                </div>
                            </div>
                        )}

                        {configTab === 'series' && (
                            <div>
                                <div className="form-group">
                                    <label>Series Config Name</label>
                                    <select
                                        value={seriesConfigName}
                                        onChange={async (e) => {
                                            setSeriesConfigName(e.target.value);
                                            await loadSeriesConfig(e.target.value);
                                        }}
                                    >
                                        {seriesConfigs.map(name => (
                                            <option key={name} value={name}>{name}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label>Series Configuration (YAML)</label>
                                    <textarea
                                        value={seriesConfig}
                                        onChange={(e) => setSeriesConfig(e.target.value)}
                                        style={{minHeight: '400px', fontFamily: 'monospace'}}
                                    />
                                </div>
                                <button onClick={saveSeriesConfig}>Save Series Config</button>
                            </div>
                        )}

                        {configTab === 'blocks' && modelType === 'dfm' && (
                            <div>
                                <div className="form-group">
                                    <label>Block Config Name</label>
                                    <select
                                        value={blockConfigName}
                                        onChange={async (e) => {
                                            setBlockConfigName(e.target.value);
                                            await loadBlockConfig(e.target.value);
                                        }}
                                    >
                                        {blockConfigs.map(name => (
                                            <option key={name} value={name}>{name}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label>Block Configuration (YAML)</label>
                                    <textarea
                                        value={blockConfig}
                                        onChange={(e) => setBlockConfig(e.target.value)}
                                        style={{minHeight: '300px', fontFamily: 'monospace'}}
                                    />
                                </div>
                                <button onClick={saveBlockConfig}>Save Block Config</button>
                            </div>
                        )}
                    </div>
                </div>

                <div className="dashboard-panel">
                    <div className="panel-header">
                        <h3 className="panel-title">Training Settings</h3>
                    </div>
                    <div className="panel-content">
                        <div className="form-group">
                            <label>Model Name (optional, auto-generated if empty)</label>
                            <input
                                type="text"
                                value={modelName}
                                onChange={(e) => setModelName(e.target.value)}
                                placeholder="Leave empty for auto-generated name"
                            />
                        </div>
                        <button onClick={startTraining} disabled={training}>
                            {training ? 'Training...' : 'Start Training'}
                        </button>
                    </div>
                </div>

                {training && (
                    <div className="dashboard-panel">
                        <div className="panel-header">
                            <h3 className="panel-title">Training Progress</h3>
                        </div>
                        <div className="panel-content">
                            <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                            </div>
                            <p>{status}</p>
                            <p>Progress: {progress}%</p>
                        </div>
                    </div>
                )}

                {error && <div className="error" style={{gridColumn: '1 / -1'}}>{error}</div>}
            </div>
        </div>
    );
}

function InferenceTab({ models }) {
    const [selectedModel, setSelectedModel] = useState('');
    const [targetSeries, setTargetSeries] = useState('');
    const [viewDate, setViewDate] = useState('');
    const [targetPeriod, setTargetPeriod] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const runInference = async () => {
        if (!selectedModel || !targetSeries || !viewDate) {
            setError('Please fill in all required fields');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/inference', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: selectedModel,
                    target_series: targetSeries,
                    view_date: viewDate,
                    target_period: targetPeriod || undefined
                })
            });

            if (!response.ok) {
                throw new Error('Inference failed');
            }

            const data = await response.json();
            setResult(data);
        } catch (error) {
            setError('Inference failed: ' + error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <div className="card">
                <h2>Run Inference</h2>
                <div className="form-group">
                    <label>Select Model</label>
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                    >
                        <option value="">-- Select a model --</option>
                        {models.map(model => (
                            <option key={model.model_name} value={model.model_name}>
                                {model.model_name} ({model.model_type})
                            </option>
                        ))}
                    </select>
                </div>
                <div className="form-group">
                    <label>Target Series</label>
                    <input
                        type="text"
                        value={targetSeries}
                        onChange={(e) => setTargetSeries(e.target.value)}
                        placeholder="e.g., gdp"
                    />
                </div>
                <div className="form-group">
                    <label>View Date</label>
                    <input
                        type="date"
                        value={viewDate}
                        onChange={(e) => setViewDate(e.target.value)}
                    />
                </div>
                <div className="form-group">
                    <label>Target Period (optional)</label>
                    <input
                        type="text"
                        value={targetPeriod}
                        onChange={(e) => setTargetPeriod(e.target.value)}
                        placeholder="e.g., 2024Q1"
                    />
                </div>
                <button onClick={runInference} disabled={loading}>
                    {loading ? 'Running...' : 'Run Inference'}
                </button>
            </div>

            {result && (
                <div className="card">
                    <h2>Results</h2>
                    <pre>{JSON.stringify(result, null, 2)}</pre>
                </div>
            )}

            {error && <div className="error">{error}</div>}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));
