import React, { useEffect, useRef } from 'react';
import Typewriter from 'typewriter-effect';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { FaLinkedin, FaGithub } from "react-icons/fa";
import { MdEmail } from "react-icons/md";

gsap.registerPlugin(ScrollTrigger);

const IconCloud = (props) => ( <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 15a4 4 0 014-4h6a5 5 0 015 5 5 5 0 01-5 5H7a4 4 0 01-4-4z" /> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M17 10l-1-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4M12 15h0.01M12 18h0.01M16 15h0.01" /> </svg> ); 

const IconSettings = (props) => ( <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.562.342 1.246.223 1.724-.064z" /> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /> </svg> ); 

const IconBell = (props) => ( <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 17h5l-1.405-2.81a1.2 1.2 0 01-.157-.75V11a7.001 7.001 0 00-14 0v2.44c0 .408-.052.793-.157 1.157L4 17h5m6 0a2 2 0 100 4 2 2 0 000-4z" /> </svg> ); 

const IconLightning = (props) => ( <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 10V3L4 14h7v7l9-11h-7z" /> </svg> ); 

const IconBrain = (props) => ( <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h.01M15 12h.01M12 18h.01M12 6h.01M12 2a4 4 0 00-4 4 4 4 0 00-4 4v4a4 4 0 004 4h12a4 4 0 004-4v-4a4 4 0 00-4-4 4 4 0 00-4-4z" /> </svg> ); 

const IconShield = (props) => ( <svg {...props} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 18h.01M12 6h.01M12 2a4 4 0 00-4 4v4a4 4 0 004 4h12a4 4 0 004-4v-4a4 4 0 00-4-4zM12 2v20M16 16l-4 4-4-4M16 8l-4-4-4 4" /> </svg> ); 

const StepArrow = () => ( <div className="flex-shrink-0 flex items-center justify-center h-16 w-8 relative overflow-hidden"> <div className="w-px h-full bg-gray-300 absolute top-0"></div> <svg className="w-5 h-5 text-gray-300 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"> 
<path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path> </svg> </div> );

const Home = () => {
  const featureRef = useRef([]);
  const workRef = useRef([]);

  useEffect(() => {
    featureRef.current.forEach((el) => {
      gsap.fromTo(el,
        { scale: 0.5, opacity: 0 },
        {
          scale: 1,
          opacity: 1,
          duration: 1,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: el,
            start: 'top 90%',
            end: 'top 10%',
            scrub:1,
            toggleActions: 'play none none none',
          },
        }
      );
    });

    // Animate How It Works Steps
    workRef.current.forEach((el) => {
      gsap.fromTo(el, 
        { scale: 0.5, opacity: 0 },
        { 
          scale: 1, 
          opacity: 1, 
          duration: 1, 
          scrollTrigger: {
            trigger: el,
            start: 'top 90%',
            end: 'top 10%',
            scrub:1,
            toggleActions: 'play none none none'
          } 
        });
    });
  }, []);
  

  return (
    <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
      
      {/* Hero Section */}
      <section className="flex flex-row md:flex-row items-center py-20 gap-10">
        <div className="w-2/3">
          <p className="text-4xl font-extrabold text-slate-900 leading-tight mb-4">
            <Typewriter
              options={{ autoStart: true, loop: true, delay: 40 }}
              onInit={(typewriter) => {
                typewriter
                  .typeString('Real-Time AML: ')
                  .typeString('<span style="color:#068ADF;">Stay Ahead</span>')
                  .typeString(' of Financial Crime')
                  .start();
              }}
            />
          </p>
          <p className="text-lg text-gray-600 mb-8">
            Our AI-powered platform provides intelligent, explainable transaction monitoring system that implements 
            real-world compliance requirements with advanced machine learning capabilities, explainable AI, and RAG-powered investigation tools.
          </p>
        </div>

        <div className="w-full md:w-1/2 flex justify-center p-4">
          <div className="relative w-full max-w-xl bg-white rounded-xl shadow-[0_20px_50px_rgba(0,0,0,0.1)] p-6 border border-gray-100">
            <div className="h-4 w-full flex items-center space-x-1.5 mb-4">
              <span className="w-3 h-3 bg-red-500 rounded-full"></span>
              <span className="w-3 h-3 bg-yellow-400 rounded-full"></span>
            </div>

            {/* Static Graph Images */}
            <div className="flex justify-between items-end mb-4 gap-6">
              <img src="/graph1.png" alt="Graph 1" className="w-1/2 h-36 object-cover rounded-lg shadow-inner" />
              <img src="/graph2.png" alt="Graph 2" className="w-1/2 h-26 object-cover rounded-lg shadow-inner" />
            </div>

            <div className="text-center text-sm text-gray-500">Real-Time Data Flow & Alerts</div>
          </div>
        </div>
      </section>

      {/* Features Section */} 
      <section id="features" className="py-10"> 
        <h2 className="text-3xl font-bold text-center text-slate-800 mb-12">Features</h2> 
        <div className="grid grid-cols-3 md:grid-cols-3 gap-8"> 
            {/* Feature Card 1: Instant Detection */} 
            <div  ref={(el) => (featureRef.current[0] = el)} className="p-8 bg-white rounded-xl shadow-[0_10px_30px_rgba(0,0,0,0.05)] border-t-2 border-transparent hover:border-blue-500 transition duration-300"> 
                <div className="flex items-center mb-2 gap-3"> 
                    <IconLightning className="w-10 h-10 text-blue-600 stroke-[1.5]" /> 
                    <h3 className="text-xl font-semibold text-slate-800">Instant Detection</h3> 
                </div> 
                <p className="text-gray-500">Flag suspicious activity in real-time with continuous monitoring and low latency scoring.</p> 
            </div>

            {/* Feature Card 2: Explainable AI */} 
            <div ref={(el) => (featureRef.current[1] = el)} className="p-8 bg-white rounded-xl shadow-[0_10px_30px_rgba(0,0,0,0.05)] border-t-2 border-transparent hover:border-blue-500 transition duration-300"> 
                <div className="flex items-center mb-2 gap-3"> 
                    <IconBrain className="w-10 h-10 text-blue-600 stroke-[1.5]" /> 
                    <h3 className="text-xl font-semibold text-slate-800">Explainable AI</h3> 
                </div> 
                <p className="text-gray-500">Understand why alerts are triggered with clear and concise insights.</p> 
            </div> 
            
            {/* Feature Card 3: Intelligent Risk Scoring */} 
            <div ref={(el) => (featureRef.current[2] = el)} className="p-8 bg-white rounded-xl shadow-[0_10px_30px_rgba(0,0,0,0.05)] border-t-2 border-transparent hover:border-blue-500 transition duration-300"> 
                <div className="flex items-center mb-2 gap-3"> 
                    <IconShield className="w-10 h-10 text-blue-600 stroke-[1.5]" /> 
                    <h3 className="text-xl font-semibold text-slate-800">Intelligent Risk Scoring</h3> 
                </div> 
                <p className="text-gray-500">Prioritize threats with dynamic, adaptive scores based on rule hits and predictive models.</p> 
            </div>
        </div>
        </section>
      
       {/* How It Works Section */} 
       <section id="how-it-works" className="py-20 mb-20"> 
        <h2 className="text-3xl font-bold text-center text-slate-800 mb-16">How it Works</h2> 
        <div className="flex justify-center items-center space-y-4 lg:space-y-0 relative"> 
        {/* Step 1: Data Ingestion */} 
        <div  ref={(el) => (workRef.current[0] = el)} className="flex flex-col items-center w-full lg:w-1/3"> 
            <div className="w-20 h-20 flex items-center justify-center rounded-full border-2 border-blue-300 text-blue-600 mb-4"> 
                <IconCloud className="w-8 h-8 stroke-1" /> 
            </div> 
            <h3 className="text-lg font-semibold text-slate-800 mb-1">1. Data Ingestion</h3> 
            <p className="text-gray-500 text-center text-sm max-w-xs">Stream transactions securely into our highly available data pipeline.</p> 
        </div> 
        {/* Connector Arrow 1 */} 
        <div className="hidden lg:block w-32 h-1 relative"> 
            <div className="absolute inset-0 border-t border-gray-300"></div> 
            <svg className="w-5 h-5 text-gray-300 absolute right-0 top-1/2 transform -translate-y-1/2 translate-x-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path></svg> 
        </div> 
        <div className="lg:hidden"> <StepArrow /> 
        </div> 
        
        {/* Step 2: Real-Time Analysis */} 
        <div  ref={(el) => (workRef.current[1] = el)} className="flex flex-col items-center w-full lg:w-1/3"> 
           <div className="w-20 h-20 flex items-center justify-center rounded-full border-2 border-blue-300 text-blue-600 mb-4"> 
              <IconSettings className="w-8 h-8 stroke-1" /> 
            </div> 
            <h3 className="text-lg font-semibold text-slate-800 mb-1">2. Real-Time Analysis</h3> 
            <p className="text-gray-500 text-center text-sm max-w-xs">Rules, ML, and referential data identify and score potential risks instantly.</p> 
        </div> 
        {/* Connector Arrow 2 */} 
        <div className="hidden lg:block w-32 h-1 relative"> 
            <div className="absolute inset-0 border-t border-gray-300"></div> 
            <svg className="w-5 h-5 text-gray-300 absolute right-0 top-1/2 transform -translate-y-1/2 translate-x-1/2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path></svg> 
        </div> 
        <div className="lg:hidden"> <StepArrow /> 
        </div> 
        
        {/* Step 3: Smart Alerts & Insights */} 
        <div  ref={(el) => (workRef.current[2] = el)} className="flex flex-col items-center w-full lg:w-1/3">
    <div className="w-20 h-20 flex items-center justify-center rounded-full border-2 border-blue-300 text-blue-600 mb-4">
        {/* Replacing Bell Icon with a generic Analysis/Chart icon for "Rescore Analysis" */}
        <svg className="w-8 h-8 stroke-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19V6l2-2m-2 15h8a2 2 0 002-2V7a2 2 0 00-2-2H9a2 2 0 00-2 2v10a2 2 0 002 2zm0 0l2 2m-2-2l2-2m-2 2h8" />
        </svg>
    </div>
    <h3 className="text-lg font-semibold text-slate-800 mb-1">3. Rescore Analysis & Insights</h3>
    <p className="text-gray-500 text-center text-sm max-w-xs">Gain deeper understanding through rescoring and detailed analysis of flagged activity.</p>
</div>
        </div> 

         {/* Call to Action Section */}
         {/* <section className="bg-slate-800 text-white py-16 text-center rounded-xl shadow-2xl mt-20 mb-10"> 
            <h2 className="text-4xl font-bold mb-6">Ready to see it in action?</h2> 
            <button className="bg-orange-600 hover:bg-orange-700 hover:scale-105 text-white font-bold py-3 px-10 rounded-lg text-lg shadow-xl transition duration-300"> Request AML-360 </button> 
        </section>  */}
        
<section className="bg-slate-800 text-white py-16 text-center rounded-xl shadow-2xl mt-20 mb-10"> 
    <h2 className="text-4xl font-bold mb-6">Ready to see it in action?</h2> 
    <button
        className="bg-orange-600 hover:bg-orange-700 hover:scale-105 text-white font-bold py-3 px-10 rounded-lg text-lg shadow-xl transition duration-300"
        onClick={() => window.location.href = "https://52b980d3-8524-4948-be8a-a82a240f3ca3-00-x02dxa3u97ad.worf.replit.dev/"}
    >
        Request AML-360
    </button> 
</section>


        <footer className="text-gray-300 py-8 mb-[-120px]">
      <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center">
        {/* Left Section */}
        <div className="text-center md:text-left mb-4 md:mb-0">
          <h2 className="text-xl font-semibold text-white">AML-360</h2>
          <p className="text-sm text-gray-400 mt-1">
            Real-time suspicious transaction monitoring and risk-based scoring.
          </p>
        </div>


        {/* Right Icons */}
        <div className="flex space-x-5 text-2xl mt-4">
          <a
            href="#"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-blue-400 transition"
          >
            <FaLinkedin />
          </a>
          <a
            href="#"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-gray-800 transition"
          >
            <FaGithub />
          </a>
          <a
            href="#"
            className="text-gray-400 hover:text-red-400 transition"
          >
            <MdEmail />
          </a>
        </div>
      </div>

      {/* Bottom Text */}
      <div className="text-center text-gray-500 text-sm mt-6 border-t border-gray-700 pt-4">
        © {new Date().getFullYear()} AML-360 | All Rights Reserved
      </div>
    </footer>

       </section>
    </main>
  );
};

export default Home;