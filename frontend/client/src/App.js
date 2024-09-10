import React, { useState } from 'react';
import './App.css';
import { Modal } from 'flowbite-react';
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
  const [isLoading, setIsLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleFileChange = (event) => {
    if (event.target.files.length > 0) {
      setScatter(event.target.files[0]); // Ensure this is a file object
    }
  };

  const handleSubmit = () => {
    setIsLoading(true); // Start loading
    setIsModalOpen(false); // Hide the modal initially

    const formData = new FormData();

    formData.append('answer1', !isNaN(parseFloat(slope)) ? parseFloat(slope) : '');
    formData.append('answer2', !isNaN(parseFloat(intercept)) ? parseFloat(intercept) : '');
    formData.append('answer3', !isNaN(parseFloat(r_squared)) ? parseFloat(r_squared) : '');
    formData.append('answer4', interpretation || ''); // Text answers
    formData.append('answer5', !isNaN(parseFloat(coefficient)) ? parseFloat(coefficient) : '');
    formData.append('scatter', scatter); // File object
    formData.append('answer7', !isNaN(parseFloat(prediction)) ? parseFloat(prediction) : '');
    formData.append('answer8', recommendation || '');

    fetch('/api/submit-answers', {
      method: 'POST',
      body: formData, // Use FormData instead of JSON.stringify
    })
      .then((response) => response.json())
      .then((data) => {
        setResponseMessage(data.message);

        setIsLoading(false);
        setIsModalOpen(true);
      })
      .catch((error) => {
        console.error('Error:', error);

        setResponseMessage('Pengajuan gagal');
        setIsLoading(false);
        setIsModalOpen(true);
      });
  };

  return (
    <>
      <nav className="bg-white border-gray-200 dark:bg-gray-900 mb-2">
        <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
          <a href="/" className="flex items-center space-x-3 rtl:space-x-reverse">
            <img src={jawatimurlogo} className="h-14" alt="Jawa Timur Logo" />
            <span className="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">Dashboard</span>
          </a>
        </div>
      </nav>

      <div className="App">
        {/* <h1>Soal 1a</h1> */}
        <div className="input-section">
          <div>
            <div className="mb-6">
              <label htmlFor="slope" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1a - Slope</label>
              <input type="text" id="slope" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                value={slope}
                onChange={(e) => setSlope(e.target.value)}
                placeholder="Isi jawaban" />
            </div>
          </div>
          <div>
            <label htmlFor="intercept" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1a - Intercept</label>
            <input type="text" id="intercept" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={intercept}
              onChange={(e) => setIntercept(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 1b</h1> */}

        <div className="input-section">
          <div>
            <label htmlFor="r_squared" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1b - R-Squared</label>
            <input type="text" id="r_squared" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={r_squared}
              onChange={(e) => setRSquared(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 1c</h1> */}

        <div className="input-section">
          <div>
            <label htmlFor="interpretation" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 1c - Interpretasi Hasil Regresi</label>
            <textarea id="message" rows="4" className="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={interpretation}
              onChange={(e) => setInterpretation(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 2</h1> */}

        <div className="input-section">
          <div>
            <label htmlFor="coefficient" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 2 - Coefficient</label>
            <input type="text" id="coefficientd" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={coefficient}
              onChange={(e) => setCoefficient(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 3</h1> */}

        <div>
          <label className="block mb-2 text-sm font-bold text-gray-900 dark:text-white text-xl " htmlFor="scatter" name="scatter">Soal 3 - Scatter Plot Penjualan vs Biaya Iklan</label>
          <input className="block w-full mb-5 text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400" id="scatter" type="file"
            accept=".png, .jpg, .jpeg"
            onChange={handleFileChange} />
            <p class="mt-1 text-sm text-gray-500 dark:text-gray-300" id="file_input_help"> (MAX. 100 MB).</p>
        </div>

        {/* <h1>Soal 4a</h1> */}

        <div className="input-section">
          <div>
            <label htmlFor="prediction" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 4a - Prediksi Penjualan</label>
            <input type="text" id="prediction" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-00 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={prediction}
              onChange={(e) => setPrediction(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <h1>Soal 4b</h1> */}

        <div className="input-section">
          <div>
            <label htmlFor="recommendation" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white text-xl">Soal 4b - Interpretasi dan Rekomendasi Hasil</label>
            <textarea id="message" rows="4" className="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
              value={recommendation}
              onChange={(e) => setRecommendation(e.target.value)}
              placeholder="Isi jawaban" />
          </div>
        </div>

        {/* <button type="button" className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800" onClick={handleSubmit} disabled={isLoading}>Submit</button> */}

        {isLoading && (
          <div className="spinner-overlay">
            <div role="status">
              <svg
                aria-hidden="true"
                className="w-8 h-8 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600"
                viewBox="0 0 100 101"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
                  fill="currentColor"
                />
                <path
                  d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
                  fill="currentFill"
                />
              </svg>
              <span className="sr-only">Loading...</span>
            </div>
          </div>
        )}


        {/* Form with Submit button */}
        {!isLoading && (
          <button type="button" className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800" onClick={handleSubmit} disabled={isLoading}>Submit</button>
        )}

        <Modal show={isModalOpen} onClose={() => setIsModalOpen(false)}>
          <Modal.Header>Hasil Jawaban</Modal.Header>
          <Modal.Body>
            <p
              dangerouslySetInnerHTML={{ __html: responseMessage }}
            ></p>
          </Modal.Body>
          <Modal.Footer>
            <button onClick={() => setIsModalOpen(false)} className="btn-close px-3 py-2 text-sm font-medium text-white bg-blue-500 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">Tutup</button>
          </Modal.Footer>
        </Modal>
      </div>
    </>

  );
}

export default App;

