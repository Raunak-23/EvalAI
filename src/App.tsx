// src/App.tsx
import React from "react";
import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Grading from "./pages/Grading";
import Analytics from "./pages/Analytics";
import Students from "./pages/Students";
import SettingsPage from "./pages/Settings";
import Help from "./pages/Help";
import Login from "./pages/Login";

import Header from "./component/Header";
import Sidebar from "./component/Sidebar";
import ProtectedRoute from "./component/ProtectedRoute";
import { AuthProvider } from "./context/AuthContext";

// Layout for authenticated pages
const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="flex min-h-screen bg-gray-50">
      <Sidebar />
      <main className="flex-1 flex flex-col">
        <Header />
        <div className="flex-1 p-6">{children}</div>
      </main>
    </div>
  );
};

// Handles animated route transitions
const AnimatedRoutes: React.FC = () => {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/login" element={<Login />} />

        <Route
          path="/"
          element={
            <ProtectedRoute>
              <AppLayout>
                <Dashboard />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/upload"
          element={
            <ProtectedRoute>
              <AppLayout>
                <Upload />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/grading"
          element={
            <ProtectedRoute>
              <AppLayout>
                <Grading />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/analytics"
          element={
            <ProtectedRoute>
              <AppLayout>
                <Analytics />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/students"
          element={
            <ProtectedRoute>
              <AppLayout>
                <Students />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <AppLayout>
                <SettingsPage />
              </AppLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/help"
          element={
            <ProtectedRoute>
              <AppLayout>
                <Help />
              </AppLayout>
            </ProtectedRoute>
          }
        />
      </Routes>
    </AnimatePresence>
  );
};

function App() {
  return (
    <Router>
      <AnimatedRoutes />
    </Router>
  );
}

export default function RootApp() {
  return (
    <AuthProvider>
      <App />
    </AuthProvider>
  );
}
