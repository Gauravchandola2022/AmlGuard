import React, { useState } from "react";
import axios from "axios";
import { countries } from "country-data"; 

// --- Icon Components ---
const IconWave = (props) => (
  <svg
    {...props}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2.5"
      d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.805A9.73 9.73 0 013 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
    />
  </svg>
);

const IconCheckmark = (props) => (
  <svg
    {...props}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="3"
      d="M5 13l4 4L19 7"
    />
  </svg>
);

const IconArrowDown = (props) => (
  <svg
    {...props}
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M19 9l-7 7-7-7"
    />
  </svg>
);

// --- Helper function to get currency ---
const getCurrencyByCountry = (countryCode) => {
  const country = countries[countryCode.toUpperCase()];
  return country && country.currencies && country.currencies.length
    ? country.currencies[0]
    : "";
};

// --- Mock Results for UI ---
const mockResults = {
  riskScore: 8,
  isSuspicious: true,
  triggeredRules: [
    {
      id: 1,
      name: "Beneficiary country (IR is Level 3 High-Risk)",
      score: "+10",
      desc: "IR is Level 3 High-Risk",
    },
    {
      id: 2,
      name: 'Suspicious keyword "crypto" (+3)',
      score: "+3",
      desc: 'Found keyword "crypto" in instruction',
    },
    {
      id: 5,
      name: "Rounded Amount (+2)",
      score: "+2",
      desc: "Amount is $100,000",
    },
  ],
  totalScoreText: "109k (mock data)",
};

// --- InputField Reusable Component ---
const InputField = ({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  readOnly = false,
}) => (
  <div className="mb-4">
    <label className="block text-sm font-medium text-gray-500 mb-1">
      {label}
    </label>
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      readOnly={readOnly}
      className={`w-full px-3 py-2 border border-gray-200 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 transition duration-150 ${
        readOnly ? "bg-gray-100 cursor-not-allowed" : ""
      }`}
    />
  </div>
);

// --- Main Component ---
const App = () => {
  const [formData, setFormData] = useState({
    transactionId: `${Date.now()}`,
    date: "2025-01-01",
    originator: { name: "", address: "", country: "" },
    beneficiary: { name: "", address: "", country: "" },
    amount: "",
    currency: "",
    paymentInstruction: "",
  });

  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [result, setResult] = useState(null);

  // --- Handle Beneficiary Country Change ---
  const handleBeneficiaryCountryChange = (e) => {
    const countryCode = e.target.value;
    const currency = getCurrencyByCountry(countryCode);

    setFormData({
      ...formData,
      beneficiary: { ...formData.beneficiary, country: countryCode },
      currency,
    });
  };

  // --- Handle Submit ---
  const handleMonitor = async () => {
    try {
      setLoading(true);
      const response = await axios.post(
        "http://localhost:8080/api/transactions",
        formData
      );
      setResult(response.data);
      setShowResults(true);
    } catch (error) {
      console.error("Error sending transaction:", error);
      alert("Failed to monitor transaction. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50 p-4">
      <div className="flex flex-col lg:flex-row gap-8 max-w-6xl w-full">
        {/* --- Left Card (Input Form) --- */}
        <div className="w-full lg:w-1/2 bg-white rounded-3xl shadow-xl p-8 md:p-12 border border-gray-100">
          <div className="flex items-center mb-10">
            <IconWave className="w-8 h-8 mr-2 text-blue-600 stroke-1" />
            <h1 className="text-3xl font-extrabold text-slate-800">
              Monitor New Transaction
            </h1>
          </div>

          {/* --- Transaction Form --- */}
          <form>
            <InputField
              label="Date"
              type="date"
              value={formData.date}
              onChange={(e) =>
                setFormData({ ...formData, date: e.target.value })
              }
            />

            <h2 className="text-lg font-semibold text-slate-700 mt-6 mb-2">
              Originator Details
            </h2>
            <InputField
              label="Originator Name"
              placeholder="Enter sender's full name"
              value={formData.originator.name}
              onChange={(e) => {
                const regex = /^[A-Za-z\s]*$/;
                if (regex.test(e.target.value)) {
                  setFormData({
                    ...formData,
                    originator: {
                      ...formData.originator,
                      name: e.target.value,
                    },
                  });
                } else {
                  alert("Only alphabets are allowed in the Originator Name.");
                }
              }}
            />

            <InputField
              label="Originator Address"
              placeholder="Enter sender's address"
              value={formData.originator.address}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  originator: {
                    ...formData.originator,
                    address: e.target.value,
                  },
                })
              }
            />

            {/* Originator Country Dropdown */}
<div className="mb-4">
  <label className="block text-sm font-medium text-gray-500 mb-1">
    Originator Country
  </label>
  <select
    value={formData.originator.country}
    onChange={(e) =>
      setFormData({
        ...formData,
        originator: { ...formData.originator, country: e.target.value },
      })
    }
    className="w-full px-3 py-2 border border-gray-200 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 transition duration-150"
  >
    <option value="">Select Country</option>
    {Object.values(countries)
      .filter((c) => c.name && c.alpha2)
      .map((c) => (
        <option key={c.alpha2} value={c.alpha2}>
          {c.name} ({c.alpha2})
        </option>
      ))}
  </select>
</div>

            <h2 className="text-lg font-semibold text-slate-700 mt-6 mb-2">
              Beneficiary Details
            </h2>

            <InputField
              label="Beneficiary Name"
              placeholder="Enter receiver's full name"
              value={formData.beneficiary.name}
              onChange={(e) => {
                const regex = /^[A-Za-z\s]*$/;
                if (regex.test(e.target.value)) {
                  setFormData({
                    ...formData,
                    beneficiary: {
                      ...formData.beneficiary,
                      name: e.target.value,
                    },
                  });
                } else {
                  alert("Only alphabets are allowed in the Beneficiary Name.");
                }
              }}
            />

            <InputField
              label="Beneficiary Address"
              placeholder="Enter receiver's address"
              value={formData.beneficiary.address}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  beneficiary: {
                    ...formData.beneficiary,
                    address: e.target.value,
                  },
                })
              }
            />

            {/* --- Beneficiary Country Dropdown --- */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-500 mb-1">
                Beneficiary Country
              </label>
              <select
                value={formData.beneficiary.country}
                onChange={handleBeneficiaryCountryChange}
                className="w-full px-3 py-2 border border-gray-200 rounded-lg shadow-sm focus:ring-blue-500 focus:border-blue-500 transition duration-150"
              >
                <option value="">Select Country</option>
                {Object.values(countries)
                  .filter((c) => c.name && c.alpha2)
                  .map((c) => (
                    <option key={c.alpha2} value={c.alpha2}>
                      {c.name} ({c.alpha2})
                    </option>
                  ))}
              </select>
            </div>

            <InputField
              label="Currency"
              placeholder="Currency"
              value={formData.currency}
              readOnly
            />

            <InputField
              label="Amount"
              type="number"
              placeholder="Enter amount"
              value={formData.amount}
              onChange={(e) =>
                setFormData({ ...formData, amount: e.target.value })
              }
            />

            <InputField
              label="Payment Instruction"
              placeholder="Payment for service"
              value={formData.paymentInstruction}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  paymentInstruction: e.target.value,
                })
              }
            />

            <button
              type="button"
              onClick={handleMonitor}
              disabled={loading}
              className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-xl shadow-lg transition duration-200"
            >
              {loading ? "Monitoring..." : "Monitor Transaction"}
            </button>
          </form>
        </div>

        {/* --- Right Card (Results) --- */}
        <div
          className={`w-full lg:w-1/2 bg-white rounded-3xl shadow-xl p-8 md:p-12 border border-gray-100 transition-opacity duration-500 ${
            showResults ? "opacity-100" : "opacity-50"
          }`}
        >
          <h2 className="text-2xl font-bold text-slate-800 mb-2">
            Risk Assessment Result
          </h2>
          <p className="text-gray-500 text-sm mb-8">
            Enter transaction details to analyze
          </p>

          <div className="flex justify-center mb-8">
            <div className="w-40 h-40 flex flex-col items-center justify-center rounded-full bg-red-600/10 border-4 border-red-600 shadow-lg relative">
              <span className="text-red-700 text-sm font-semibold mb-1">
                Risk Score:
              </span>
              <span className="text-red-700 text-4xl font-extrabold">
                {mockResults.riskScore}
              </span>
            </div>
          </div>

          <div className="flex items-center justify-center mb-10">
            <IconCheckmark className="w-8 h-8 text-green-500 mr-3 stroke-[3]" />
            <span className="text-2xl font-bold text-green-700">
              Suspicious Transaction Flagged
            </span>
          </div>

          <h3 className="text-lg font-semibold text-slate-800 mb-4">
            Triggered Rules
          </h3>

          <div className="space-y-3">
            {mockResults.triggeredRules.map((rule) => (
              <div
                key={rule.id}
                className="flex items-start p-3 rounded-lg border border-gray-200 shadow-sm"
              >
                <IconArrowDown className="w-5 h-5 text-blue-500 mr-3 transform rotate-90 stroke-2 mt-0.5" />
                <div className="flex-grow">
                  <p className="text-sm font-medium text-slate-800">
                    <span className="text-blue-600 mr-1">Rule #{rule.id}:</span>
                    {rule.name}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">{rule.desc}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 pt-4 border-t border-gray-200">
            <p className="text-sm font-bold text-slate-800 mb-6">
              Total Score:{" "}
              <span className="text-red-600">{mockResults.totalScoreText}</span>
            </p>
            <button className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 rounded-xl shadow-lg transition duration-200">
              Open Chatbot for Explanation
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
