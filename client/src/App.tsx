import { useState } from "react";
import Sidebar from "./component/Sidebar";
import TeacherInfo from "./component/TeacherInfo";
import ClassSelector from "./component/ClassSelector";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const App = () => {
  const [selectedClass, setSelectedClass] = useState("");
  const [mode, setMode] = useState<"single" | "bulk">("single");
  const [files, setFiles] = useState<File[]>([]);
  const [processing, setProcessing] = useState(false);
  const [resultsVisible, setResultsVisible] = useState(false);

  // Process overview states
  const [showProcessOverview, setShowProcessOverview] = useState(false);
  const [completedCount, setCompletedCount] = useState(0);
  const [processingCount, setProcessingCount] = useState(0);
  const [failedCount, setFailedCount] = useState(0);

  const classes = ["SJT 424", "PRP 122", "SJT 607", "SMV 201"];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const selectedFiles = Array.from(e.target.files);
    setFiles(mode === "single" ? selectedFiles.slice(0, 1) : selectedFiles);
  };

  const handleUpload = () => {
    if (files.length === 0) return;
    setProcessing(true);
    setResultsVisible(false);
    setShowProcessOverview(true);

    // Reset counts
    setCompletedCount(0);
    setFailedCount(0);
    setProcessingCount(files.length);

    // Fake OCR delay
    setTimeout(() => {
      setProcessing(false);
      setResultsVisible(true);
      setCompletedCount(files.length); 
      setFailedCount(0);
      setProcessingCount(0); // remove yellow line
    }, 3000);
  };

  // Sample chart data
  const chartData = {
    labels: ["Thinking", "Handwriting", "Logic", "Application"],
    datasets: [
      {
        label: "Performance (%)",
        data: [85, 78, 82, 75],
        backgroundColor: "rgba(37, 99, 235, 0.7)",
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "top" as const },
      title: { display: true, text: "Performance Breakdown" },
    },
  };

  return (
    <div className="flex min-h-screen bg-gray-50">
      <Sidebar />
      <main className="flex-1 p-6">
        <header className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Teacher Dashboard</h1>
        </header>

        <ClassSelector
          classes={classes}
          selectedClass={selectedClass}
          setSelectedClass={setSelectedClass}
        />

        {selectedClass && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Teacher Info + Process Overview */}
            <div className="lg:col-span-1 space-y-6">
              <TeacherInfo />

              {/* Process Overview Box */}
              {showProcessOverview && (
                <div className="p-4 bg-white rounded-xl shadow-md space-y-2">
                  <h2 className="text-lg font-semibold mb-2">Process Overview</h2>
                  <p className="flex items-center gap-2 text-green-600">
                    🟢 Completed: {completedCount}
                  </p>
                  {processingCount > 0 && (
                    <p className="flex items-center gap-2 text-yellow-600">
                      🟡 Processing: {processingCount}
                    </p>
                  )}
                  <p className="flex items-center gap-2 text-red-600">
                    🔴 Failed: {failedCount}
                  </p>
                </div>
              )}
            </div>

            {/* Main Box Container */}
            <div className="lg:col-span-3 bg-white p-6 rounded-xl shadow-lg space-y-6">
              {/* Upload Panel */}
              <div>
                <h2 className="text-lg font-semibold mb-4">Upload Student Papers</h2>
                <div className="flex gap-4 mb-4">
                  <button
                    className={`px-4 py-2 rounded-md ${
                      mode === "single" ? "bg-blue-600 text-white" : "bg-gray-200"
                    }`}
                    onClick={() => setMode("single")}
                  >
                    Single Student
                  </button>
                  <button
                    className={`px-4 py-2 rounded-md ${
                      mode === "bulk" ? "bg-blue-600 text-white" : "bg-gray-200"
                    }`}
                    onClick={() => setMode("bulk")}
                  >
                    Whole Class
                  </button>
                </div>

                <input
                  type="file"
                  multiple={mode === "bulk"}
                  onChange={handleFileChange}
                  className="block w-full border rounded-md p-2 mb-4"
                  accept=".pdf"
                />

                <button
                  onClick={handleUpload}
                  disabled={processing || files.length === 0}
                  className={`w-full py-2 rounded-md text-white ${
                    processing || files.length === 0
                      ? "bg-gray-400"
                      : "bg-green-600 hover:bg-green-700"
                  }`}
                >
                  {processing ? "Processing..." : "Upload & Process"}
                </button>
              </div>

              {/* Results Panel */}
              {resultsVisible && (
                <div>
                  <h2 className="text-lg font-semibold mb-4">Results & Analytics</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-4 bg-gray-50 rounded-md">
                      <p>
                        <strong>Marks:</strong> 72 / 100
                      </p>
                      <p>
                        <strong>Strengths:</strong> Algebra, Derivatives
                      </p>
                      <p>
                        <strong>Weaknesses:</strong> Integration, Word Problems
                      </p>
                      {mode === "bulk" && (
                        <p>
                          <strong>Class Average:</strong> 68
                        </p>
                      )}
                    </div>
                    <div className="p-4 bg-gray-50 rounded-md">
                      <Bar data={chartData} options={chartOptions} />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
