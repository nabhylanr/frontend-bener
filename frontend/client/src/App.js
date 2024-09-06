import React, { useState } from 'react';
import './App.css';

function App() {
  const [slope, setSlope] = useState('');
  const [intercept, setIntercept] = useState('');
  const [r_squared, setRSquared] = useState('');
  const [interpretation, setInterpretation] = useState('');
  const [coefficient, setCoefficient] = useState('');
  const [scatter, setScatter] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [recommendation, setRecommendation] = useState('');
  const [responseMessage, setResponseMessage] = useState('');
  const [responseStatus, setResponseStatus] = useState('');
  

  const handleSubmit = () => {
    fetch('/api/submit-answers', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        answer1: parseFloat(slope),
        answer2: parseFloat(intercept),
        answer3: parseFloat(r_squared),
        answer4: interpretation,
        answer5: parseFloat(coefficient),
        answer6: scatter,
        answer7: parseFloat(prediction),
        answer8: recommendation,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        setResponseStatus(data.status);
        setResponseMessage(data.message);
      })
      .catch((error) => console.error('Error:', error));
  };

  //handle file input changes
  const handleImageChange = (e) => {
    const selectedFile = e.target.files[0];
    
    if (selectedFile) {
      const fileType = selectedFile.type;
      if (fileType === 'image/jpeg' || fileType === 'image/png') {
        // set the file state if valid
        setScatter(selectedFile);
        console.log('File sudah benar:', selectedFile);
      } else {
        alert('Harap unggah file dengan format jpg, jpeg dan png.');
        e.target.value = '';
      }
    }
  };

  return (
    <div className="App">
      <h1>Soal 1a</h1>

      <div className="input-section">
        <div>
          <label htmlFor="slope">Slope</label>
          <input
            type="text"
            id="slope"
            value={slope}
            onChange={(e) => setSlope(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="intercept">Intercept</label>
          <input
            type="text"
            id="intercept"
            value={intercept}
            onChange={(e) => setIntercept(e.target.value)}
          />
        </div>
      </div>

      <h1>Soal 1b</h1>

      <div className="input-section">
        <div>
          <label htmlFor="r_squared">R-Squared</label>
          <input
            type="text"
            id="r_squared"
            value={r_squared}
            onChange={(e) => setRSquared(e.target.value)}
          />
        </div>
      </div>

      <h1>Soal 1c</h1>

      <div className="input-section">
        <div>
          <label htmlFor="interpretation">Interpretasi Hasil Regresi</label>
          <textarea
            id="interpretation"
            value={interpretation}
            onChange={(e) => setInterpretation(e.target.value)}
          />
        </div>
      </div>

      <h1>Soal 2</h1>

      <div className="input-section">
        <div>
          <label htmlFor="coefficient">Koefisien Korelasi antara Penjualan dan Biaya Iklan</label>
          <input
            type="text"
            id="coefficient"
            value={coefficient}
            onChange={(e) => setCoefficient(e.target.value)}
          />
        </div>
      </div>

    <h1>Soal 3</h1>
      
      <div className="input-section">
        <div>
          <label htmlFor="scatter">Scatter Plot Penjualan vs Biaya Iklan</label>
          <input
            type="file"
            id="scatter"
            accept=".png, .jpg, .jpeg"
            onChange={handleImageChange} 
          />
        </div>
      </div>

      <h1>Soal 4a</h1>

      <div className="input-section">
        <div>
          <label htmlFor="prediction">Prediksi Penjualan</label>
          <input
            type="text"
            id="prediction"
            value={prediction}
            onChange={(e) => setPrediction(e.target.value)}
          />
        </div>
      </div>

      <h1>Soal 4b</h1>

      <div className="input-section">
        <div>
          <label htmlFor="recommendation">Interpretasi dan Rekomendasi Hasil</label>
          <textarea
            id="recommendation"
            value={recommendation}
            onChange={(e) => setRecommendation(e.target.value)}
          />
        </div>
      </div>

      <button onClick={handleSubmit}>Submit</button>

        <div className="response">
          {responseMessage && (
            <p className={responseStatus === 'success' ? 'success' : 'error'}>
              {responseMessage}
            </p>
          )}
        </div>
    </div>
    
  );
}

export default App;

