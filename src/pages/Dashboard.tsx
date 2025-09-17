// src/pages/Dashboard.tsx
import React from "react";
import { motion } from "framer-motion";
import TeacherInfo from "../component/TeacherInfo";

const Dashboard: React.FC = () => {
  const recentActivity = [
    "Evaluation For VL2025260101664",
    "Graded Quiz For VL2025260101666",
  ];
  const pendingEvaluations = [
    { student: "Bharath Rajiv R", assignment: "review 2 (will give full marks)" },
  ];

  return (
    <div className="flex flex-col flex-1 p-6 bg-gray-50 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="flex flex-col flex-1"
      >
        <h1 className="text-2xl font-bold mb-6">Teacher Dashboard</h1>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Column */}
          <div className="lg:col-span-1 flex flex-col space-y-6">
            {/* Teacher Info Card */}
            <div className="bg-white p-6 rounded-xl shadow-xl border border-gray-200">
              <TeacherInfo />
            </div>

            {/* Recent Activity Card */}
            <div className="bg-white p-6 rounded-xl shadow-xl border border-gray-200">
              <h2 className="text-lg font-semibold mb-2">Recent Activity</h2>
              <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                {recentActivity.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Pending Evaluations Card */}
            <div className="bg-white p-6 rounded-xl shadow-xl border border-gray-200">
              <h2 className="text-lg font-semibold mb-2">Pending Evaluations</h2>
              <ul className="text-sm text-gray-700 space-y-1">
                {pendingEvaluations.map((item, idx) => (
                  <li key={idx}>
                    <span className="font-medium">{item.student}</span>: {item.assignment}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 bg-white p-6 rounded-xl shadow-xl border border-gray-200 max-w-full">
            <p className="text-gray-600">
              Analytics and grading overview will appear here.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
