import React from "react";

const Sidebar = () => {
  return (
    <div className="w-64 bg-white shadow-lg h-screen p-4">
      <h2 className="text-xl font-bold mb-4">Menu</h2>
      <ul className="space-y-2">
        <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Dashboard</li>
        <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Settings</li>
        <li className="text-gray-700 hover:text-blue-600 cursor-pointer">Help</li>
      </ul>
    </div>
  );
};

export default Sidebar;
