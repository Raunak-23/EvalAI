import React from "react";
import { motion } from "framer-motion";
import { HelpCircle } from "lucide-react"; // Icon for Help

export default function Help() {
  return (
    <div className="flex flex-col flex-1 p-6 bg-gray-50 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 bg-white rounded-xl shadow-md"
      >
        <div className="flex items-center gap-2 mb-4">
          <HelpCircle className="text-yellow-600" />
          <h2 className="text-xl font-bold">Help & Support</h2>
        </div>
        <p className="text-gray-600">
          This page is under development. Help documentation and support resources will be available soon.
        </p>
      </motion.div>
    </div>
  );
}
