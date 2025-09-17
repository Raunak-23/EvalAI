// src/pages/Upload.tsx
import React, { useState } from "react";
import Select from "react-select";
import { motion } from "framer-motion";
import TeacherInfo from "../component/TeacherInfo";

const Upload: React.FC = () => {
  const [mode, setMode] = useState<"single" | "bulk">("single");
  const [selectedClass, setSelectedClass] = useState<any>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [processing, setProcessing] = useState(false);
  const [resultsVisible, setResultsVisible] = useState(false);

  const [completed, setCompleted] = useState(0);
  const [failed, setFailed] = useState(0);
  const [processingCount, setProcessingCount] = useState(0);

  const classes = [
    { value: "VL2025260101634", label: "VL2025260101634" },
    { value: "VL2025260101642", label: "VL2025260101642" },
    { value: "VL2025260101621", label: "VL2025260101621" },
    { value: "VL2025260101689", label: "VL2025260101689" },
  ];

  const handleFiles = (selectedFiles: FileList | null) => {
    if (!selectedFiles) return;
    const fileArray = Array.from(selectedFiles);
    setFiles(mode === "single" ? fileArray.slice(0, 1) : fileArray);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const handleUpload = () => {
    if (!selectedClass || files.length === 0) return;

    setProcessing(true);
    setResultsVisible(false);

    // Reset counts
    setCompleted(0);
    setProcessingCount(files.length);
    setFailed(0);

    // Simulate processing delay
    setTimeout(() => {
      setCompleted(files.length);
      setProcessingCount(0);
      setFailed(0);
      setProcessing(false);
      setResultsVisible(true);
    }, 2000);
  };

  return (
    <div className="flex flex-col flex-1 p-6 bg-gray-50 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex flex-col lg:grid lg:grid-cols-4 gap-6"
      >
        {/* Left Column */}
        <div className="lg:col-span-1 space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <TeacherInfo />
          </motion.div>

          {/* Processing Overview Box */}
          {(processing || resultsVisible) && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="p-4 bg-white rounded-xl shadow-lg border border-gray-200"
            >
              <h3 className="text-lg font-semibold mb-2">Processing Overview</h3>
              <p className="text-sm text-gray-600 mb-2">
                Queue Status: <span className="font-semibold">{files.length} file{files.length > 1 ? "s" : ""}</span>
              </p>
              <div className="w-full bg-gray-200 h-2 rounded mb-4">
                <div
                  className="bg-blue-600 h-2 rounded"
                  style={{ width: `${(completed / files.length) * 100}%` }}
                ></div>
              </div>
              <ul className="space-y-1 text-sm">
                <li>✅ Completed: {completed}</li>
                <li>🔄 Processing: {processingCount}</li>
                <li>❌ Failed: {failed}</li>
              </ul>
            </motion.div>
          )}
        </div>

        {/* Right Column: Upload Panel */}
        <div className="lg:col-span-3 space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="bg-white p-6 shadow-lg rounded-xl border border-gray-200"
          >
            <h2 className="text-lg font-semibold mb-4">Upload Student Papers</h2>

            <Select
              options={classes}
              value={selectedClass}
              onChange={setSelectedClass}
              placeholder="Select Class"
              isClearable
              className="mb-4"
            />

            <div className="flex gap-4 mb-4">
              <button
                className={`px-4 py-2 rounded-md ${mode === "single" ? "bg-blue-600 text-white" : "bg-gray-200"}`}
                onClick={() => setMode("single")}
              >
                Single Student
              </button>
              <button
                className={`px-4 py-2 rounded-md ${mode === "bulk" ? "bg-blue-600 text-white" : "bg-gray-200"}`}
                onClick={() => setMode("bulk")}
              >
                Whole Class
              </button>
            </div>

            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="w-full border-2 border-dashed border-gray-300 rounded-md p-6 text-center cursor-pointer hover:bg-gray-50"
              onClick={() => document.getElementById("fileInput")?.click()}
            >
              <p className="text-gray-600 mb-2">
                Drag & drop your {mode === "single" ? "file" : "files"} here, or click to browse
              </p>
              <input
                id="fileInput"
                type="file"
                multiple={mode === "bulk"}
                onChange={(e) => handleFiles(e.target.files)}
                className="hidden"
                accept=".pdf"
              />
              {files.length > 0 && (
                <ul className="mt-2 text-sm text-gray-700 text-left">
                  {files.map((file, idx) => (
                    <li key={idx}>• {file.name}</li>
                  ))}
                </ul>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={processing || files.length === 0 || !selectedClass}
              className={`w-full py-2 rounded-md text-white mt-4 ${
                processing || files.length === 0 || !selectedClass
                  ? "bg-gray-400"
                  : "bg-green-600 hover:bg-green-700"
              }`}
            >
              {processing ? "Processing..." : "Upload & Process"}
            </button>
          </motion.div>

          {/* Results Card */}
          {resultsVisible && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="p-4 bg-gray-50 rounded-xl shadow-lg border border-gray-200"
            >
              <h2 className="text-lg font-semibold mb-2">Results & Analytics</h2>
              <p><strong>Class:</strong> {selectedClass?.label}</p>
              <p><strong>Marks:</strong> 72 / 100</p>
              <p><strong>Strengths:</strong> Algebra, Derivatives</p>
              <p><strong>Weaknesses:</strong> Integration, Word Problems</p>
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default Upload;
