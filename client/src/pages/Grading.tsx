import React from "react";
import { motion } from "framer-motion";
import { AlertCircle } from "lucide-react";

export default function Grading() {
  return (
    <div className="flex flex-col flex-1 p-6 bg-gray-50 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white rounded-xl shadow-md"
      >
        <div className="flex items-center gap-2 mb-4">
          <AlertCircle className="text-red-600" />
          <h2 className="text-xl font-bold">Grading</h2>
        </div>
        <p className="text-gray-600">
          This is where grading-related features will be shown.
        </p>
      </motion.div>
    </div>
  );
}
