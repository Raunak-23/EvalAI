import React from "react";

interface ClassSelectorProps {
  classes: string[];
  selectedClass: string;
  setSelectedClass: (cls: string) => void;
}

const ClassSelector: React.FC<ClassSelectorProps> = ({ classes, selectedClass, setSelectedClass }) => {
  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-gray-700 mb-2">Select Class</label>
      <select
        value={selectedClass}
        onChange={(e) => setSelectedClass(e.target.value)}
        className="w-full p-2 border rounded-md"
      >
        <option value="">-- Choose a Class --</option>
        {classes.map((cls) => (
          <option key={cls} value={cls}>
            {cls}
          </option>
        ))}
      </select>
    </div>
  );
};

export default ClassSelector;
