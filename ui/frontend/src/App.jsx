import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Monitor from './monitor.jsx';
import Home from './home.jsx'; 

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50 font-inter">
        {/* Header Section */}
        <header className="sticky top-0 z-50 bg-white mx-auto px-4 sm:px-6 lg:px-8 border-b border-gray-100 shadow-lg">
          <div className="flex justify-between items-center py-4">
          <div className='flex justify-center'>
          <img 
            src="logo.png" 
            alt="AML 360 Logo" 
            className="w-9 h-9 mr-3 object-contain mt-[-4px]" 
          />
           AML-360
          </div>
            <nav className="flex items-center space-x-6">
              <Link to="/Home" className="text-sm font-medium text-gray-600 hover:text-blue-600 transition duration-150">Home</Link>
              <Link to="/Monitor" className="text-sm font-medium text-gray-600 hover:text-blue-600 transition duration-150">Create Transaction</Link>
              <button className="bg-gradient-to-r from-blue-400 to-purple-500 text-white font-semibold py-2 px-5 rounded-full hover:from-blue-600 hover:to-purple-700 transition-all duration-300 shadow-md">
                Sign In
              </button>
            </nav>
          </div>
        </header>

        {/* Routes */}
        <Routes>
          <Route path="/Home" element={<Home />} />
          <Route path="/Monitor" element={<Monitor />} />
          <Route path="/" element={<Home />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
