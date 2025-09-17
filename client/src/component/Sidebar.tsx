// src/component/Sidebar.tsx
import React from "react";
import { NavLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import {
  Home,
  UploadCloud,
  FileText,
  BarChart2,
  Users,
  Settings,
  HelpCircle,
  LogOut,
} from "lucide-react";

const Sidebar = () => {
  const { logout } = useAuth();

  const menuItems = [
    { name: "Dashboard", to: "/", icon: <Home className="w-5 h-5 mr-2" /> },
    { name: "Upload", to: "/upload", icon: <UploadCloud className="w-5 h-5 mr-2" /> },
    { name: "Grading", to: "/grading", icon: <FileText className="w-5 h-5 mr-2" /> },
    { name: "Analytics", to: "/analytics", icon: <BarChart2 className="w-5 h-5 mr-2" /> },
    { name: "Students", to: "/students", icon: <Users className="w-5 h-5 mr-2" /> },
    { name: "Settings", to: "/settings", icon: <Settings className="w-5 h-5 mr-2" /> },
    { name: "Help", to: "/help", icon: <HelpCircle className="w-5 h-5 mr-2" /> },
  ];

  return (
    <div className="bg-white border-r border-gray-300 flex flex-col justify-between min-h-screen" style={{ width: "200px" }}>
      {/* Header */}
      <div className="p-4 border-b border-gray-300 flex items-center justify-center" style={{ height: "80px" }}>
        <h1 className="text-3xl font-bold text-blue-600">EvalAI</h1>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-2 space-y-2">
        {menuItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center px-4 py-2 rounded hover:bg-gray-100 ${
                isActive ? "bg-gray-200 font-semibold" : ""
              }`
            }
          >
            {item.icon}
            {item.name}
          </NavLink>
        ))}
      </nav>

      {/* Logout */}
      <div className="p-4 border-t border-gray-300">
        <button
          onClick={logout}
          className="w-full flex items-center justify-center bg-red-600 text-white py-2 rounded hover:bg-red-700"
        >
          <LogOut className="w-5 h-5 mr-2" /> Logout
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
