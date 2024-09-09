import React, { useState } from 'react';
import './App.css';
import jawatimurlogo from './images/jawatimurlogo.png';

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

    <>
      <nav class="bg-white border-gray-200 dark:bg-gray-900 mb-2">
        <div class="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
          <a href="/" class="flex items-center space-x-3 rtl:space-x-reverse">
            <img src={jawatimurlogo} class="h-14" alt="Jawa Timur Logo" />
            <span class="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">Dashboard</span>
          </a>
        </div>
      </nav>



      <div className="App">
        {/* <h1>Soal 1a</h1> */}
        <div className="input-section">
          <div>
            <div class="mb-6">
              <label for="slope" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1a - Slope</label>
              <input type="text" id="slope" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                value={slope}
                onChange={(e) => setSlope(e.target.value)}
                placeholder="Isi jawaban" />
            </div>
          </div>
          <div>
            <label for="intercept" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1a - Intercept</label>
            <input type="text" id="intercept" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={intercept}
              onChange={(e) => setIntercept(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 1b</h1> */}

        <div className="input-section">
          <div>
            <label for="r_squared" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1b - R-Squared</label>
            <input type="text" id="r_squared" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={r_squared}
              onChange={(e) => setRSquared(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 1c</h1> */}

        <div className="input-section">
          <div>
            <label for="interpretation" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1c - Interpretasi Hasil Regresi</label>
            <textarea id="message" rows="4" class="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={interpretation}
              onChange={(e) => setInterpretation(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 2</h1> */}

        <div className="input-section">
          <div>
            <label for="coefficient" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 2 - Coefficient</label>
            <input type="text" id="coefficientd" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={coefficient}
              onChange={(e) => setCoefficient(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 3</h1> */}

        <div>
          <label class="block mb-2 text-sm font-bold text-gray-900 dark:text-white text-xl " for="scatter">Soal 3 - Scatter Plot Penjualan vs Biaya Iklan</label>
          <input class="block w-full mb-5 text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400" id="scatter" type="file"
            accept=".png, .jpg, .jpeg"
            onChange={handleImageChange} />
        </div>

        {/* <h1>Soal 4a</h1> */}

        <div className="input-section">
          <div>
            <label for="prediction" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 4a - Prediksi Penjualan</label>
            <input type="text" id="prediction" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={prediction}
              onChange={(e) => setPrediction(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 4b</h1> */}

        <div className="input-section">
          <div>
            <label for="recommendation" class="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 4b - Interpretasi dan Rekomendasi Hasil</label>
            <textarea id="message" rows="4" class="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={recommendation}
              onChange={(e) => setRecommendation(e.target.value)}
              placeholder="Isi jawaban" />

          </div>
        </div>
        <button type="button" class="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800" onClick={handleSubmit}>Submit</button>

        <div className="response">
          {responseMessage && (
            <p
              className={responseStatus === 'success' ? 'success' : 'error'}
              dangerouslySetInnerHTML={{ __html: responseMessage }}
            ></p>
          )}
        </div>
      </div>

    </>
  );
}

export default App;

