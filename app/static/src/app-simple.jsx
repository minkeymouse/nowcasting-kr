const { useState, useEffect } = React;

// Experiment Selection Screen
function ExperimentSelection({ experiments, onSelect }) {
    const [searchQuery, setSearchQuery] = useState('');

    const filtered = experiments.filter(exp => 
        exp.experiment_id.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div style={{padding: '40px', maxWidth: '800px', margin: '0 auto'}}>
            <h1>Select Experiment</h1>
            <input
                type="text"
                placeholder="Search experiments..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{width: '100%', padding: '10px', marginBottom: '20px', fontSize: '16px'}}
            />
            <div style={{display: 'grid', gap: '10px'}}>
                {filtered.map(exp => (
                    <div
                        key={exp.experiment_id}
                        onClick={() => onSelect(exp.experiment_id)}
                        style={{
                            padding: '15px',
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            background: '#f9f9f9'
                        }}
                        onMouseEnter={(e) => e.target.style.background = '#f0f0f0'}
                        onMouseLeave={(e) => e.target.style.background = '#f9f9f9'}
                    >
                        <strong>{exp.experiment_id}</strong>
                    </div>
                ))}
            </div>
            {filtered.length === 0 && (
                <p style={{textAlign: 'center', color: '#666'}}>No experiments found</p>
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
            const response = await fetch('/api/upload/data', {
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
        <div style={{padding: '20px', background: 'white', borderRadius: '8px', marginBottom: '20px'}}>
            <h3>Upload Data</h3>
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
                    <label>Date Column Name</label>
                    <input
                        type="text"
                        value={dateColumn}
                        onChange={(e) => setDateColumn(e.target.value)}
                        placeholder="e.g., date"
                        style={{width: '100%', padding: '8px'}}
                    />
                </div>
                <div>
                    <label>Date Format</label>
                    <input
                        type="text"
                        value={dateFormat}
                        onChange={(e) => setDateFormat(e.target.value)}
                        placeholder="e.g., YYYY-MM-DD"
                        style={{width: '100%', padding: '8px'}}
                    />
                </div>
                <div>
                    <label>Save As (filename)</label>
                    <input
                        type="text"
                        value={filename}
                        onChange={(e) => setFilename(e.target.value)}
                        placeholder="e.g., sample_data.csv"
                        style={{width: '100%', padding: '8px'}}
                    />
                </div>
                <button
                    onClick={handleUpload}
                    disabled={uploading || !file}
                    style={{
                        padding: '10px 20px',
                        background: '#007bff',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: uploading ? 'not-allowed' : 'pointer'
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
            const response = await fetch('/api/upload/config', {
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
        <div style={{padding: '20px', background: 'white', borderRadius: '8px', marginBottom: '20px'}}>
            <h3>Upload Config</h3>
            <p style={{color: '#666', fontSize: '14px', marginBottom: '15px'}}>
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
                        padding: '10px 20px',
                        background: '#28a745',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: uploading ? 'not-allowed' : 'pointer'
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
            return item.series_id?.toLowerCase().includes(search) ||
                   item.series_name?.toLowerCase().includes(search);
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
                <h2>{title}</h2>
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
                                padding: '12px',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                background: '#f9f9f9'
                            }}
                            onMouseEnter={(e) => e.target.style.background = '#f0f0f0'}
                            onMouseLeave={(e) => e.target.style.background = '#f9f9f9'}
                        >
                            {type === 'model' ? (
                                <div>
                                    <strong>{item.model_name || item.model_type}</strong>
                                    {item.model_type && <span style={{color: '#666', marginLeft: '10px'}}>({item.model_type})</span>}
                                </div>
                            ) : (
                                <div>
                                    <strong>{item.series_id || item.series_name}</strong>
                                    {item.series_name && <div style={{color: '#666', fontSize: '14px'}}>{item.series_name}</div>}
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
                        marginTop: '15px',
                        padding: '8px 16px',
                        background: '#6c757d',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
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
    const [availableModels, setAvailableModels] = useState([]);
    const [availableSeries, setAvailableSeries] = useState([]);

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
            // Extract models and series from config
            if (data.config?.model_type) {
                setModels([{ model_type: data.config.model_type }]);
            }
            if (data.config?.preprocess?.series) {
                setSeries(data.config.preprocess.series);
            }
        } catch (error) {
            console.error('Failed to load experiment:', error);
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
            const response = await fetch('/api/config/series');
            const data = await response.json();
            setAvailableSeries(Array.isArray(data) ? data : []);
        } catch (error) {
            console.error('Failed to load series:', error);
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

    return (
        <div style={{display: 'flex', height: '100vh'}}>
            {/* Left Sidebar */}
            <div style={{
                width: '300px',
                background: '#f5f5f5',
                padding: '20px',
                borderRight: '1px solid #ddd',
                overflow: 'auto'
            }}>
                <h2 style={{marginTop: 0}}>Configuration</h2>
                
                {/* Model Section */}
                <div style={{marginBottom: '20px'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px'}}>
                        <strong>Model</strong>
                        <button
                            onClick={() => setShowModelBrowser(true)}
                            style={{
                                padding: '4px 8px',
                                background: '#007bff',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                        >
                            +
                        </button>
                    </div>
                    {models.length > 0 ? (
                        <div style={{padding: '8px', background: 'white', borderRadius: '4px'}}>
                            {models.map((m, idx) => (
                                <div key={idx}>{m.model_type}</div>
                            ))}
                        </div>
                    ) : (
                        <div style={{color: '#999', fontSize: '14px'}}>No model selected</div>
                    )}
                </div>

                {/* Series Section */}
                <div style={{marginBottom: '20px'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px'}}>
                        <strong>Series</strong>
                        <button
                            onClick={() => setShowSeriesBrowser(true)}
                            style={{
                                padding: '4px 8px',
                                background: '#28a745',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                        >
                            +
                        </button>
                    </div>
                    {series.length > 0 ? (
                        <div style={{maxHeight: '300px', overflow: 'auto'}}>
                            {series.map((s, idx) => (
                                <div key={idx} style={{padding: '8px', background: 'white', borderRadius: '4px', marginBottom: '5px'}}>
                                    <div><strong>{s.series_id}</strong></div>
                                    {s.series_name && <div style={{fontSize: '12px', color: '#666'}}>{s.series_name}</div>}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div style={{color: '#999', fontSize: '14px'}}>No series added</div>
                    )}
                </div>
            </div>

            {/* Main Content */}
            <div style={{flex: 1, padding: '20px', overflow: 'auto'}}>
                <h1>Dashboard - {experimentId}</h1>
                
                <UploadData />
                <UploadConfig />
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
        </div>
    );
}

// Main App
function App() {
    const [experiments, setExperiments] = useState([]);
    const [selectedExperiment, setSelectedExperiment] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
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

