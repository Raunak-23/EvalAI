import React from "react";
import { motion } from "framer-motion";
import { User } from "lucide-react"; // optional icon

export default function Students() {
  return (
    <div className="flex flex-col flex-1 p-6 bg-gray-50 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white rounded-xl shadow-md"
      >
        <div className="flex items-center gap-2 mb-4">
          {/* Remove icon if lucide-react not installed */}
          <User className="text-blue-600" /> 
          <h2 className="text-xl font-bold">Student Management</h2>
        </div>
        <p className="text-gray-600">
          This page is under development. Student management features will be available soon.
        </p>
      </motion.div>
    </div>
  );
}
