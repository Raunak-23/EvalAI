import React from "react";
import { User, Mail } from "lucide-react";

const TeacherInfo = () => {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-lg font-semibold mb-4">Teacher Information</h2>
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <User size={16} className="text-blue-600" />
          <span>Name: <strong>DHIVYAA C R</strong></span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <Mail size={16} className="text-green-600" />
          <span>Email: <strong>dhivyaa@gmail.com</strong></span>
        </div>
      </div>
    </div>
  );
};

export default TeacherInfo;


