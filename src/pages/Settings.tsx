import React from "react";
import { motion } from "framer-motion";
// import { Settings } from "lucide-react"; // Optional: uncomment if lucide-react is installed

export default function SettingsPage() {
  return (
    <div className="flex flex-col flex-1 p-6 bg-gray-50 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white rounded-xl shadow-md"
      >
        <div className="flex items-center gap-2 mb-4">
          {/* Use an icon if you have lucide-react installed */}
          {/* <Settings className="text-green-600" /> */}
          <h2 className="text-xl font-bold">Settings & Configuration</h2>
        </div>
        <p className="text-gray-600">
          This page is under development. System settings and configuration options will be available soon.
        </p>
      </motion.div>
    </div>
  );
}


