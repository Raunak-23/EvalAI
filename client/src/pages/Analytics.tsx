import React from "react";
import { motion } from "framer-motion";
import { BarChart3 } from "lucide-react";

export default function Analytics() {
  return (
    <div className="flex flex-col flex-1 p-6 bg-gray-50 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white rounded-xl shadow-md"
      >
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="text-blue-600" />
          <h2 className="text-xl font-bold">Analytics</h2>
        </div>
        <p className="text-gray-600">
          This is where analytics and performance breakdown will be displayed.
        </p>
      </motion.div>
    </div>
  );
}
