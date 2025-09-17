// src/component/Header.tsx
import React, { useState } from "react";
import { Search, User } from "lucide-react"; // Using User icon as a teacher placeholder

const Header: React.FC = () => {
  const [query, setQuery] = useState("");

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Searching for:", query);
  };

  return (
    <header
      className="w-full bg-white border-b border-gray-200 shadow-sm flex justify-between items-center px-4"
      style={{ height: "80px" }} // Match sidebar header height
    >
      {/* Left - Website Title */}
     {/*<h1 className="text-xl font-bold text-blue-600">EvalAI</h1>*/}

      {/* Center - Search Bar */}
      <form
        onSubmit={handleSearch}
        className="flex items-center border rounded-lg overflow-hidden w-1/2"
      >
        <input
          type="text"
          placeholder="Search students, classes, etc..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 px-4 py-2 outline-none text-gray-700"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 flex items-center"
        >
          <Search className="w-5 h-5" />
        </button>
      </form>

      {/* Right - Teacher Icon */}
      <div className="flex items-center gap-2 text-gray-700">
        <User className="w-6 h-6 text-blue-600" />
        <span className="text-sm font-medium"> Welcome,Teacher</span>
      </div>
    </header>
  );
};

export default Header;
