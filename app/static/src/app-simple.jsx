const { useState, useEffect } = React;

// Experiment Selection Screen
function ExperimentSelection({ experiments, onSelect }) {
    const [searchQuery, setSearchQuery] = useState('');

    const filtered = experiments.filter(exp => 
        exp.experiment_id.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div style={{
            padding: '80px 40px', 
            maxWidth: '900px', 
            margin: '0 auto', 
            background: '#ffffff', 
            minHeight: '100vh'
        }}>
            <div style={{
                marginBottom: '48px',
                paddingBottom: '32px',
                borderBottom: '2px solid #E3E3E3'
            }}>
                <h1 style={{
                    color: '#1B3C53', 
                    marginBottom: '0', 
                    marginTop: '0',
                    fontSize: '40px', 
                    fontWeight: '700',
                    letterSpacing: '-0.8px',
                    lineHeight: '1.2'
                }}>Select Experiment</h1>
                <p style={{
                    color: '#456882',
                    fontSize: '16px',
                    marginTop: '12px',
                    marginBottom: '0',
                    fontWeight: '400'
                }}>Choose an experiment to configure and run</p>
            </div>
            <div style={{
                marginBottom: '32px',
                position: 'relative'
            }}>
                <input
                    type="text"
                    placeholder="Search experiments..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{
                        width: '100%', 
                        padding: '16px 20px', 
                        fontSize: '16px',
                        border: '2px solid #E3E3E3',
                        borderRadius: '8px',
                        background: 'white',
                        color: '#1B3C53',
                        boxShadow: '0 2px 8px rgba(27, 60, 83, 0.06)',
                        transition: 'all 0.2s ease'
                    }}
                    onFocus={(e) => {
                        e.target.style.borderColor = '#1B3C53';
                        e.target.style.boxShadow = '0 4px 12px rgba(27, 60, 83, 0.12)';
                    }}
                    onBlur={(e) => {
                        e.target.style.borderColor = '#E3E3E3';
                        e.target.style.boxShadow = '0 2px 8px rgba(27, 60, 83, 0.06)';
                    }}
                />
            </div>
            <div style={{display: 'grid', gap: '14px'}}>
                {filtered.map(exp => (
                    <div
                        key={exp.experiment_id}
                        onClick={() => onSelect(exp.experiment_id)}
                        style={{
                            padding: '20px 24px',
                            border: '1px solid #E3E3E3',
                            borderRadius: '8px',
                            cursor: 'pointer',
                            background: 'white',
                            color: '#1B3C53',
                            transition: 'all 0.2s ease',
                            boxShadow: '0 2px 6px rgba(27, 60, 83, 0.08)'
                        }}
                        onMouseEnter={(e) => {
                            e.target.style.background = '#f8f9fa';
                            e.target.style.borderColor = '#1B3C53';
                            e.target.style.transform = 'translateY(-2px)';
                            e.target.style.boxShadow = '0 4px 12px rgba(27, 60, 83, 0.15)';
                        }}
                        onMouseLeave={(e) => {
                            e.target.style.background = 'white';
                            e.target.style.borderColor = '#E3E3E3';
                            e.target.style.transform = 'translateY(0)';
                            e.target.style.boxShadow = '0 2px 6px rgba(27, 60, 83, 0.08)';
                        }}
                    >
                        <strong style={{color: '#1B3C53', fontSize: '16px', fontWeight: '600'}}>{exp.experiment_id}</strong>
                        {exp.model_type && (
                            <span style={{
                                marginLeft: '12px',
                                padding: '5px 12px',
                                background: '#1B3C53',
                                color: 'white',
                                borderRadius: '12px',
                                fontSize: '11px',
                                fontWeight: '600',
                                letterSpacing: '0.5px',
                                textTransform: 'uppercase'
                            }}>
                                {exp.model_type}
                            </span>
                        )}
                    </div>
                ))}
            </div>
            {filtered.length === 0 && (
                <div style={{
                    textAlign: 'center', 
                    padding: '60px 20px',
                    marginTop: '40px'
                }}>
                    <p style={{
                        color: '#456882', 
                        fontSize: '18px', 
                        marginBottom: '8px',
                        fontWeight: '500'
                    }}>No experiments found</p>
                    <p style={{
                        color: '#6c757d', 
                        fontSize: '14px',
                        fontWeight: '400'
                    }}>Try adjusting your search query</p>
                </div>
            )}
        </div>
    );
}

// Upload Data Component
function UploadData({ onUpload }) {
    const [file, setFile] = useState(null);
    const [dateColumn, setDateColumn] = useState('date');
    const [dateFormat, setDateFormat] = useState('YYYY-MM-DD');
    const [filename, setFilename] = useState('sample_data.csv');
    const [uploading, setUploading] = useState(false);

    const handleUpload = async () => {
        if (!file) {
            alert('Please select a file');
            return;
        }

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('date_column', dateColumn);
        formData.append('date_format', dateFormat);
        formData.append('filename', filename);

        try {
            const response = await fetch('/api/data', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            const data = await response.json();
            alert('Data uploaded successfully!');
            if (onUpload) onUpload();
        } catch (error) {
            alert('Upload failed: ' + error.message);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div style={{
            padding: '28px', 
            background: 'white', 
            borderRadius: '8px', 
            marginBottom: '24px',
            boxShadow: '0 2px 8px rgba(27, 60, 83, 0.08)',
            border: '1px solid #E3E3E3'
        }}>
            <h3 style={{color: '#1B3C53', marginBottom: '24px', fontSize: '20px', fontWeight: '600'}}>Upload Data</h3>
            <div style={{display: 'grid', gap: '18px'}}>
                <div>
                    <label style={{display: 'block', marginBottom: '10px', color: '#1B3C53', fontWeight: '600', fontSize: '14px'}}>CSV File</label>
                    <input
                        type="file"
                        accept=".csv"
                        onChange={(e) => setFile(e.target.files[0])}
                        style={{
                            width: '100%', 
                            padding: '12px 14px',
                            border: '1px solid #E3E3E3',
                            borderRadius: '6px',
                            background: 'white',
                            color: '#1B3C53',
                            fontSize: '14px'
                        }}
                    />
                </div>
                <div>
                    <label style={{display: 'block', marginBottom: '10px', color: '#1B3C53', fontWeight: '600', fontSize: '14px'}}>Date Column Name</label>
                    <input
                        type="text"
                        value={dateColumn}
                        onChange={(e) => setDateColumn(e.target.value)}
                        placeholder="e.g., date"
                        style={{
                            width: '100%', 
                            padding: '12px 14px',
                            border: '1px solid #E3E3E3',
                            borderRadius: '6px',
                            background: 'white',
                            color: '#1B3C53',
                            fontSize: '14px'
                        }}
                    />
                </div>
                <div>
                    <label style={{display: 'block', marginBottom: '10px', color: '#1B3C53', fontWeight: '600', fontSize: '14px'}}>Date Format</label>
                    <input
                        type="text"
                        value={dateFormat}
                        onChange={(e) => setDateFormat(e.target.value)}
                        placeholder="e.g., YYYY-MM-DD"
                        style={{
                            width: '100%', 
                            padding: '12px 14px',
                            border: '1px solid #E3E3E3',
                            borderRadius: '6px',
                            background: 'white',
                            color: '#1B3C53',
                            fontSize: '14px'
                        }}
                    />
                </div>
                <div>
                    <label style={{display: 'block', marginBottom: '10px', color: '#1B3C53', fontWeight: '600', fontSize: '14px'}}>Save As (filename)</label>
                    <input
                        type="text"
                        value={filename}
                        onChange={(e) => setFilename(e.target.value)}
                        placeholder="e.g., sample_data.csv"
                        style={{
                            width: '100%', 
                            padding: '12px 14px',
                            border: '1px solid #E3E3E3',
                            borderRadius: '6px',
                            background: 'white',
                            color: '#1B3C53',
                            fontSize: '14px'
                        }}
                    />
                </div>
                <button
                    onClick={handleUpload}
                    disabled={uploading || !file}
                    style={{
                        padding: '14px 28px',
                        background: uploading || !file ? '#E3E3E3' : '#1B3C53',
                        color: uploading || !file ? '#6c757d' : 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: uploading || !file ? 'not-allowed' : 'pointer',
                        fontSize: '14px',
                        fontWeight: '600',
                        transition: 'all 0.2s ease',
                        width: '100%',
                        marginTop: '8px'
                    }}
                    onMouseEnter={(e) => {
                        if (!uploading && file) {
                            e.target.style.background = '#234C6A';
                            e.target.style.transform = 'translateY(-1px)';
                            e.target.style.boxShadow = '0 4px 8px rgba(27, 60, 83, 0.2)';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (!uploading && file) {
                            e.target.style.background = '#1B3C53';
                            e.target.style.transform = 'translateY(0)';
                            e.target.style.boxShadow = 'none';
                        }
                    }}
                >
                    {uploading ? 'Uploading...' : 'Upload Data'}
                </button>
            </div>
        </div>
    );
}

// Upload Config Component
function UploadConfig({ onUpload }) {
    const [file, setFile] = useState(null);
    const [filename, setFilename] = useState('metadata.csv');
    const [uploading, setUploading] = useState(false);

    const handleUpload = async () => {
        if (!file) {
            alert('Please select a file');
            return;
        }

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('filename', filename);

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            const data = await response.json();
            alert('Config uploaded successfully!');
            if (onUpload) onUpload();
        } catch (error) {
            alert('Upload failed: ' + error.message);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div style={{
            padding: '24px', 
            background: 'white', 
            borderRadius: '8px', 
            marginBottom: '20px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
            <h3 style={{color: '#1B3C53', marginBottom: '20px', fontSize: '20px', fontWeight: '600'}}>Upload Config</h3>
            <p style={{color: '#6c757d', fontSize: '14px', marginBottom: '20px', lineHeight: '1.6'}}>
                CSV should have columns: series_name, series_description, frequency, release
            </p>
            <div style={{display: 'grid', gap: '15px'}}>
                <div>
                    <label>CSV File</label>
                    <input
                        type="file"
                        accept=".csv"
                        onChange={(e) => setFile(e.target.files[0])}
                        style={{width: '100%', padding: '8px'}}
                    />
                </div>
                <div>
                    <label>Save As (filename)</label>
                    <input
                        type="text"
                        value={filename}
                        onChange={(e) => setFilename(e.target.value)}
                        placeholder="e.g., metadata.csv"
                        style={{width: '100%', padding: '8px'}}
                    />
                </div>
                <button
                    onClick={handleUpload}
                    disabled={uploading || !file}
                    style={{
                        padding: '14px 28px',
                        background: uploading || !file ? '#E3E3E3' : '#1B3C53',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: uploading || !file ? 'not-allowed' : 'pointer',
                        fontSize: '14px',
                        fontWeight: '600',
                        transition: 'all 0.2s ease',
                        width: '100%',
                        marginTop: '8px'
                    }}
                    onMouseEnter={(e) => {
                        if (!uploading && file) {
                            e.target.style.background = '#234C6A';
                            e.target.style.transform = 'translateY(-1px)';
                            e.target.style.boxShadow = '0 4px 8px rgba(27, 60, 83, 0.2)';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (!uploading && file) {
                            e.target.style.background = '#1B3C53';
                            e.target.style.transform = 'translateY(0)';
                            e.target.style.boxShadow = 'none';
                        }
                    }}
                >
                    {uploading ? 'Uploading...' : 'Upload Config'}
                </button>
            </div>
        </div>
    );
}

// Model/Series Browser Modal
function BrowserModal({ title, items, onSelect, onClose, type }) {
    const [searchQuery, setSearchQuery] = useState('');

    const filtered = items.filter(item => {
        const search = searchQuery.toLowerCase();
        if (type === 'model') {
            return item.model_name?.toLowerCase().includes(search) || 
                   item.model_type?.toLowerCase().includes(search);
        } else {
            // Handle both object format and string format
            const seriesId = item.series_id || item.series_name || item || '';
            return seriesId.toLowerCase().includes(search);
        }
    });

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
        }} onClick={onClose}>
            <div style={{
                background: 'white',
                borderRadius: '8px',
                padding: '20px',
                maxWidth: '600px',
                width: '90%',
                maxHeight: '80vh',
                overflow: 'auto'
            }} onClick={(e) => e.stopPropagation()}>
                <h2 style={{color: '#1B3C53', fontSize: '20px', fontWeight: '600', marginBottom: '20px'}}>{title}</h2>
                <input
                    type="text"
                    placeholder="Search..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{width: '100%', padding: '10px', marginBottom: '15px'}}
                />
                <div style={{display: 'grid', gap: '10px', maxHeight: '400px', overflow: 'auto'}}>
                    {filtered.map((item, idx) => (
                        <div
                            key={idx}
                            onClick={() => {
                                onSelect(item);
                                onClose();
                            }}
                            style={{
                                padding: '18px 20px',
                                border: '1px solid #E3E3E3',
                                borderRadius: '8px',
                                cursor: 'pointer',
                                background: 'white',
                                transition: 'all 0.2s ease',
                                boxShadow: '0 2px 6px rgba(27, 60, 83, 0.08)'
                            }}
                            onMouseEnter={(e) => {
                                e.target.style.background = '#f8f9fa';
                                e.target.style.borderColor = '#1B3C53';
                                e.target.style.transform = 'translateY(-2px)';
                                e.target.style.boxShadow = '0 4px 12px rgba(27, 60, 83, 0.15)';
                            }}
                            onMouseLeave={(e) => {
                                e.target.style.background = 'white';
                                e.target.style.borderColor = '#E3E3E3';
                                e.target.style.transform = 'translateY(0)';
                                e.target.style.boxShadow = '0 2px 6px rgba(27, 60, 83, 0.08)';
                            }}
                        >
                            {type === 'model' ? (
                                <div>
                                    <strong style={{color: '#1B3C53', fontSize: '16px', fontWeight: '600'}}>{item.model_name || item.model_type || 'Unknown'}</strong>
                                    {item.model_type && (
                                        <span style={{
                                            marginLeft: '12px',
                                            padding: '5px 12px',
                                            background: '#1B3C53',
                                            color: 'white',
                                            borderRadius: '12px',
                                            fontSize: '11px',
                                            fontWeight: '600',
                                            letterSpacing: '0.5px',
                                            textTransform: 'uppercase'
                                        }}>
                                            {item.model_type}
                                        </span>
                                    )}
                                </div>
                            ) : (
                                <div>
                                    <strong style={{color: '#1B3C53', fontSize: '16px', fontWeight: '600'}}>{item.series_id || item.series_name || item || 'Unknown'}</strong>
                                    {item.series_name && item.series_name !== item.series_id && (
                                        <div style={{fontSize: '14px', color: '#6c757d', marginTop: '4px'}}>{item.series_name}</div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
                {filtered.length === 0 && (
                    <p style={{textAlign: 'center', color: '#666', padding: '20px'}}>No items found</p>
                )}
                <button
                    onClick={onClose}
                    style={{
                        marginTop: '20px',
                        padding: '12px 24px',
                        background: '#1B3C53',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: '600',
                        width: '100%',
                        transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                        e.target.style.background = '#234C6A';
                        e.target.style.transform = 'translateY(-1px)';
                    }}
                    onMouseLeave={(e) => {
                        e.target.style.background = '#1B3C53';
                        e.target.style.transform = 'translateY(0)';
                    }}
                >
                    Close
                </button>
            </div>
        </div>
    );
}

// Main Dashboard
function Dashboard({ experimentId }) {
    const [models, setModels] = useState([]);
    const [series, setSeries] = useState([]);
    const [showModelBrowser, setShowModelBrowser] = useState(false);
    const [showSeriesBrowser, setShowSeriesBrowser] = useState(false);
    const [showUploadDataModal, setShowUploadDataModal] = useState(false);
    const [showUploadConfigModal, setShowUploadConfigModal] = useState(false);
    const [availableModels, setAvailableModels] = useState([]);
    const [availableSeries, setAvailableSeries] = useState([]);
    const [currentJobId, setCurrentJobId] = useState(null);
    const [jobStatus, setJobStatus] = useState(null);
    const [experimentLogs, setExperimentLogs] = useState([]);
    const [isRunning, setIsRunning] = useState(false);

    useEffect(() => {
        loadExperimentData();
        loadAvailableModels();
        loadAvailableSeries();
    }, [experimentId]);

    const loadExperimentData = async () => {
        if (!experimentId) return;
        try {
            const response = await fetch(`/api/experiment/${experimentId}/unified`);
            const data = await response.json();
            console.log('Loaded experiment data:', data);
            
            // Extract models and series from config
            if (data.config?.model_type) {
                setModels([{ model_type: data.config.model_type }]);
            }
            
            // Load series from preprocess.series
            const seriesData = data.config?.preprocess?.series;
            console.log('Raw series data from API:', seriesData);
            console.log('Is array?', Array.isArray(seriesData));
            console.log('Length:', seriesData?.length);
            
            if (seriesData && Array.isArray(seriesData) && seriesData.length > 0) {
                console.log('Setting series:', seriesData);
                setSeries(seriesData);
            } else {
                console.log('No series found in config, setting empty array');
                setSeries([]);
            }
        } catch (error) {
            console.error('Failed to load experiment:', error);
            setSeries([]);
        }
    };

    const loadAvailableModels = async () => {
        try {
            const response = await fetch('/api/models');
            setAvailableModels(await response.json());
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    };

    const loadAvailableSeries = async () => {
        try {
            const response = await fetch('/api/series');
            const data = await response.json();
            // API returns array of strings, convert to objects
            const seriesObjects = Array.isArray(data) 
                ? data.map(seriesId => ({ series_id: seriesId, series_name: seriesId }))
                : [];
            setAvailableSeries(seriesObjects);
        } catch (error) {
            console.error('Failed to load series:', error);
            setAvailableSeries([]);
        }
    };

    const handleAddModel = async (model) => {
        if (!experimentId) return;
        try {
            const response = await fetch(`/api/experiment/${experimentId}/unified`);
            const data = await response.json();
            data.config = data.config || {};
            data.config.model_type = model.model_type || model.model_name;
            
            await fetch(`/api/experiment/${experimentId}/unified`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: data.config })
            });
            
            loadExperimentData();
        } catch (error) {
            alert('Failed to add model: ' + error.message);
        }
    };

    const handleAddSeries = async (seriesItem) => {
        if (!experimentId) return;
        try {
            const response = await fetch(`/api/experiment/${experimentId}/unified`);
            const data = await response.json();
            data.config = data.config || {};
            data.config.preprocess = data.config.preprocess || {};
            data.config.preprocess.series = data.config.preprocess.series || [];
            
            // Check if already exists
            const exists = data.config.preprocess.series.some(s => 
                s.series_id === seriesItem.series_id || s.series_id === seriesItem.series_name
            );
            
            if (exists) {
                alert('Series already added');
                return;
            }
            
            data.config.preprocess.series.push({
                series_id: seriesItem.series_id || seriesItem.series_name,
                series_name: seriesItem.series_name,
                frequency: seriesItem.frequency || 'm',
                transformation: seriesItem.transformation || 'chg'
            });
            
            await fetch(`/api/experiment/${experimentId}/unified`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config: data.config })
            });
            
            loadExperimentData();
        } catch (error) {
            alert('Failed to add series: ' + error.message);
        }
    };

    const handleRunExperiment = async () => {
        if (!experimentId) {
            alert('Please select an experiment');
            return;
        }
        if (models.length === 0) {
            alert('Please select a model');
            return;
        }
        if (series.length === 0) {
            alert('Please add at least one series');
            return;
        }

        setIsRunning(true);
        setCurrentJobId(null);
        setJobStatus(null);
        setExperimentLogs([]);
        addLog('Starting experiment...');

        try {
            const response = await fetch('/api/experiment/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    experiment_id: experimentId,
                    model_name: models[0].model_type
                })
            });

            if (!response.ok) throw new Error('Failed to start experiment');
            const data = await response.json();
            setCurrentJobId(data.job_id);
            addLog(`Experiment started with job ID: ${data.job_id}`);
            
            // Start polling for status
            pollJobStatus(data.job_id);
        } catch (error) {
            addLog(`Error: ${error.message}`, 'error');
            setIsRunning(false);
            alert('Failed to run experiment: ' + error.message);
        }
    };

    const pollJobStatus = async (jobId) => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/api/experiment/status/${jobId}`);
                if (!response.ok) throw new Error('Failed to get status');
                const status = await response.json();
                setJobStatus(status);
                
                if (status.status === 'running') {
                    addLog('Experiment is running...');
                } else if (status.status === 'completed') {
                    addLog('Experiment completed successfully!', 'success');
                    setIsRunning(false);
                    clearInterval(interval);
                } else if (status.status === 'failed') {
                    addLog(`Experiment failed: ${status.error || 'Unknown error'}`, 'error');
                    setIsRunning(false);
                    clearInterval(interval);
                }
            } catch (error) {
                addLog(`Error checking status: ${error.message}`, 'error');
                clearInterval(interval);
                setIsRunning(false);
            }
        }, 2000); // Poll every 2 seconds
    };

    const addLog = (message, type = 'info') => {
        const timestamp = new Date().toLocaleTimeString();
        setExperimentLogs(prev => [...prev, { timestamp, message, type }]);
    };

    return (
        <div style={{display: 'flex', height: '100vh'}}>
            {/* Left Sidebar */}
            <div style={{
                width: '300px',
                background: '#ffffff',
                padding: '24px',
                borderRight: '1px solid #E3E3E3',
                overflow: 'auto',
                boxShadow: '2px 0 8px rgba(27, 60, 83, 0.05)',
                display: 'flex',
                flexDirection: 'column'
            }}>
                <h2 style={{marginTop: 0, color: '#1B3C53', fontSize: '22px', fontWeight: '700', marginBottom: '24px'}}>Configuration</h2>
                
                {/* Model Section */}
                <div style={{marginBottom: '20px'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px'}}>
                        <strong style={{color: '#1B3C53', fontSize: '16px', fontWeight: '600'}}>Model</strong>
                        <button
                            onClick={() => setShowModelBrowser(true)}
                            style={{
                                padding: '8px 14px',
                                background: '#1B3C53',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '16px',
                                fontWeight: '600',
                                transition: 'all 0.2s ease',
                                width: '36px',
                                height: '36px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}
                            onMouseEnter={(e) => {
                                e.target.style.background = '#234C6A';
                                e.target.style.transform = 'scale(1.05)';
                            }}
                            onMouseLeave={(e) => {
                                e.target.style.background = '#1B3C53';
                                e.target.style.transform = 'scale(1)';
                            }}
                        >
                            +
                        </button>
                    </div>
                    {models.length > 0 ? (
                        <div style={{
                            padding: '14px', 
                            background: 'white', 
                            borderRadius: '6px',
                            border: '1px solid #E3E3E3',
                            boxShadow: '0 2px 4px rgba(27, 60, 83, 0.05)'
                        }}>
                            {models.map((m, idx) => (
                                <div key={idx} style={{
                                    color: '#1B3C53', 
                                    fontSize: '14px', 
                                    fontWeight: '600',
                                    padding: '8px',
                                    background: '#f8f9fa',
                                    borderRadius: '4px'
                                }}>{m.model_type}</div>
                            ))}
                        </div>
                    ) : (
                        <div style={{
                            padding: '14px', 
                            background: '#f8f9fa', 
                            borderRadius: '6px',
                            border: '1px solid #E3E3E3',
                            color: '#6c757d', 
                            fontSize: '14px',
                            textAlign: 'center'
                        }}>No model selected</div>
                    )}
                </div>

                {/* Series Section */}
                <div style={{marginBottom: '20px'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px'}}>
                        <strong style={{color: '#1B3C53', fontSize: '16px', fontWeight: '600'}}>Series</strong>
                        <button
                            onClick={() => setShowSeriesBrowser(true)}
                            style={{
                                padding: '8px 14px',
                                background: '#1B3C53',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '16px',
                                fontWeight: '600',
                                transition: 'all 0.2s ease',
                                width: '36px',
                                height: '36px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}
                            onMouseEnter={(e) => {
                                e.target.style.background = '#234C6A';
                                e.target.style.transform = 'scale(1.05)';
                            }}
                            onMouseLeave={(e) => {
                                e.target.style.background = '#1B3C53';
                                e.target.style.transform = 'scale(1)';
                            }}
                        >
                            +
                        </button>
                    </div>
                    {series.length > 0 ? (
                        <div style={{maxHeight: '300px', overflow: 'auto'}}>
                            {series.map((s, idx) => (
                                <div key={idx} style={{
                                    padding: '14px', 
                                    background: 'white', 
                                    borderRadius: '6px', 
                                    marginBottom: '10px',
                                    border: '1px solid #E3E3E3',
                                    boxShadow: '0 2px 4px rgba(27, 60, 83, 0.05)'
                                }}>
                                    <div style={{color: '#1B3C53', fontSize: '14px', fontWeight: '600', marginBottom: '4px'}}><strong>{s.series_id}</strong></div>
                                    {s.series_name && <div style={{fontSize: '13px', color: '#6c757d', marginTop: '4px'}}>{s.series_name}</div>}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div style={{
                            padding: '14px', 
                            background: '#f8f9fa', 
                            borderRadius: '6px',
                            border: '1px solid #E3E3E3',
                            color: '#6c757d', 
                            fontSize: '14px',
                            textAlign: 'center'
                        }}>No series added</div>
                    )}
                </div>

                {/* RUN Button */}
                <div style={{
                    marginTop: 'auto',
                    paddingTop: '24px',
                    borderTop: '1px solid #E3E3E3'
                }}>
                    <button
                        onClick={handleRunExperiment}
                        disabled={isRunning || models.length === 0 || series.length === 0}
                        style={{
                            width: '100%',
                            padding: '14px 28px',
                            background: isRunning || models.length === 0 || series.length === 0 ? '#E3E3E3' : '#1B3C53',
                            color: isRunning || models.length === 0 || series.length === 0 ? '#6c757d' : 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: isRunning || models.length === 0 || series.length === 0 ? 'not-allowed' : 'pointer',
                            fontSize: '16px',
                            fontWeight: '700',
                            transition: 'all 0.2s ease',
                            textTransform: 'uppercase',
                            letterSpacing: '1px'
                        }}
                        onMouseEnter={(e) => {
                            if (!isRunning && models.length > 0 && series.length > 0) {
                                e.target.style.background = '#234C6A';
                                e.target.style.transform = 'translateY(-1px)';
                                e.target.style.boxShadow = '0 4px 8px rgba(27, 60, 83, 0.2)';
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (!isRunning && models.length > 0 && series.length > 0) {
                                e.target.style.background = '#1B3C53';
                                e.target.style.transform = 'translateY(0)';
                                e.target.style.boxShadow = 'none';
                            }
                        }}
                    >
                        {isRunning ? 'Running...' : 'RUN'}
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div style={{
                flex: 1, 
                padding: '40px', 
                overflow: 'auto',
                background: '#ffffff',
                minHeight: '100vh',
                display: 'flex',
                flexDirection: 'column'
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '32px'
                }}>
                    <h1 style={{
                        color: '#1B3C53', 
                        fontSize: '32px', 
                        fontWeight: '700', 
                        margin: 0,
                        letterSpacing: '-0.5px'
                    }}>Dashboard - {experimentId}</h1>
                    <div style={{display: 'flex', gap: '12px'}}>
                        <button
                            onClick={() => setShowUploadDataModal(true)}
                            style={{
                                padding: '10px 20px',
                                background: '#1B3C53',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '14px',
                                fontWeight: '600',
                                transition: 'all 0.2s ease'
                            }}
                            onMouseEnter={(e) => {
                                e.target.style.background = '#234C6A';
                            }}
                            onMouseLeave={(e) => {
                                e.target.style.background = '#1B3C53';
                            }}
                        >
                            Upload Data
                        </button>
                        <button
                            onClick={() => setShowUploadConfigModal(true)}
                            style={{
                                padding: '10px 20px',
                                background: '#234C6A',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                fontSize: '14px',
                                fontWeight: '600',
                                transition: 'all 0.2s ease'
                            }}
                            onMouseEnter={(e) => {
                                e.target.style.background = '#1B3C53';
                            }}
                            onMouseLeave={(e) => {
                                e.target.style.background = '#234C6A';
                            }}
                        >
                            Upload Config
                        </button>
                    </div>
                </div>

                {/* Experiment Logs Dashboard */}
                {currentJobId || jobStatus ? (
                    <div style={{
                        flex: 1,
                        background: 'white',
                        borderRadius: '8px',
                        border: '1px solid #E3E3E3',
                        boxShadow: '0 2px 8px rgba(27, 60, 83, 0.08)',
                        padding: '24px',
                        display: 'flex',
                        flexDirection: 'column'
                    }}>
                        <h2 style={{
                            color: '#1B3C53',
                            fontSize: '24px',
                            fontWeight: '700',
                            marginBottom: '20px'
                        }}>Experiment Logs</h2>
                        
                        {jobStatus && (
                            <div style={{
                                marginBottom: '20px',
                                padding: '16px',
                                background: jobStatus.status === 'completed' ? '#d4edda' : 
                                          jobStatus.status === 'failed' ? '#f8d7da' : '#d1ecf1',
                                borderRadius: '6px',
                                border: `1px solid ${jobStatus.status === 'completed' ? '#c3e6cb' : 
                                                      jobStatus.status === 'failed' ? '#f5c6cb' : '#bee5eb'}`
                            }}>
                                <div style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    marginBottom: '8px'
                                }}>
                                    <strong style={{color: '#1B3C53', fontSize: '16px'}}>Status: {jobStatus.status}</strong>
                                    <span style={{color: '#456882', fontSize: '14px'}}>Job ID: {jobStatus.job_id}</span>
                                </div>
                                {jobStatus.error && (
                                    <div style={{color: '#BD271E', fontSize: '14px', marginTop: '8px'}}>
                                        Error: {jobStatus.error}
                                    </div>
                                )}
                            </div>
                        )}

                        <div style={{
                            flex: 1,
                            background: '#1B3C53',
                            borderRadius: '6px',
                            padding: '16px',
                            overflow: 'auto',
                            fontFamily: 'monospace',
                            fontSize: '13px',
                            color: '#E3E3E3',
                            minHeight: '400px',
                            maxHeight: '600px'
                        }}>
                            {experimentLogs.length === 0 ? (
                                <div style={{color: '#456882'}}>No logs yet...</div>
                            ) : (
                                experimentLogs.map((log, idx) => (
                                    <div key={idx} style={{
                                        marginBottom: '8px',
                                        color: log.type === 'error' ? '#ff6b6b' : 
                                              log.type === 'success' ? '#51cf66' : '#E3E3E3'
                                    }}>
                                        <span style={{color: '#456882'}}>[{log.timestamp}]</span> {log.message}
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                ) : (
                    <div style={{
                        flex: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: '#f8f9fa',
                        borderRadius: '8px',
                        border: '1px solid #E3E3E3',
                        color: '#456882',
                        fontSize: '18px'
                    }}>
                        Click RUN to start experiment
                    </div>
                )}
            </div>

            {/* Modals */}
            {showModelBrowser && (
                <BrowserModal
                    title="Browse Models"
                    items={availableModels}
                    onSelect={handleAddModel}
                    onClose={() => setShowModelBrowser(false)}
                    type="model"
                />
            )}
            {showSeriesBrowser && (
                <BrowserModal
                    title="Browse Series"
                    items={availableSeries}
                    onSelect={handleAddSeries}
                    onClose={() => setShowSeriesBrowser(false)}
                    type="series"
                />
            )}
            {showUploadDataModal && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: 'rgba(0,0,0,0.5)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1000
                }} onClick={() => setShowUploadDataModal(false)}>
                    <div style={{
                        background: 'white',
                        borderRadius: '8px',
                        padding: '28px',
                        maxWidth: '600px',
                        width: '90%',
                        maxHeight: '80vh',
                        overflow: 'auto'
                    }} onClick={(e) => e.stopPropagation()}>
                        <UploadData onUpload={() => {
                            setShowUploadDataModal(false);
                        }} />
                    </div>
                </div>
            )}
            {showUploadConfigModal && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: 'rgba(0,0,0,0.5)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1000
                }} onClick={() => setShowUploadConfigModal(false)}>
                    <div style={{
                        background: 'white',
                        borderRadius: '8px',
                        padding: '28px',
                        maxWidth: '600px',
                        width: '90%',
                        maxHeight: '80vh',
                        overflow: 'auto'
                    }} onClick={(e) => e.stopPropagation()}>
                        <UploadConfig onUpload={() => {
                            setShowUploadConfigModal(false);
                        }} />
                    </div>
                </div>
            )}
        </div>
    );
}

// Main App
function App() {
    const [experiments, setExperiments] = useState([]);
    const [selectedExperiment, setSelectedExperiment] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check URL parameters for experiment ID
        const urlParams = new URLSearchParams(window.location.search);
        const experimentParam = urlParams.get('experiment');
        if (experimentParam) {
            setSelectedExperiment(experimentParam);
        }
        
        loadExperiments();
    }, []);

    const loadExperiments = async () => {
        try {
            const response = await fetch('/api/experiments');
            const data = await response.json();
            setExperiments(data);
        } catch (error) {
            console.error('Failed to load experiments:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return <div style={{padding: '40px', textAlign: 'center'}}>Loading...</div>;
    }

    if (!selectedExperiment) {
        return (
            <ExperimentSelection
                experiments={experiments}
                onSelect={setSelectedExperiment}
            />
        );
    }

    return (
        <Dashboard experimentId={selectedExperiment} />
    );
}

ReactDOM.render(<App />, document.getElementById('root'));

