import { useState } from 'react';
import PropTypes from 'prop-types';

export const PEPRiskManualInput = ({ onDataChange, initialData = {} }) => {
  const [formData, setFormData] = useState({
    age_years: initialData.age_years || '',
    gender_male: initialData.gender_male ?? null,
    bmi: initialData.bmi || '',
    cholecystectomy: initialData.cholecystectomy ?? null,
    history_of_pep: initialData.history_of_pep ?? null,
    hx_of_recurrent_pancreatitis: initialData.hx_of_recurrent_pancreatitis ?? null,
    sod: initialData.sod ?? null,
    pancreo_biliary_malignancy: initialData.pancreo_biliary_malignancy ?? null,
    trainee_involvement: initialData.trainee_involvement ?? null,
  });

  const handleChange = (field, value) => {
    const newData = { ...formData, [field]: value };
    setFormData(newData);

    // Convert to proper types and filter out empty strings
    const processedData = {};
    Object.keys(newData).forEach(key => {
      const val = newData[key];
      if (val === '' || val === null) {
        return; // Skip empty/null values
      }

      if (key === 'age_years') {
        const parsed = parseInt(val);
        if (!isNaN(parsed)) processedData[key] = parsed;
      } else if (key === 'bmi') {
        const parsed = parseFloat(val);
        if (!isNaN(parsed)) processedData[key] = parsed;
      } else if (typeof val === 'boolean') {
        processedData[key] = val;
      }
    });

    onDataChange(processedData);
  };

  const BooleanField = ({ label, field, tooltip, option_yes, option_no }) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
        {tooltip && (
          <span className="ml-1 text-gray-400 text-xs" title={tooltip}>ⓘ</span>
        )}
      </label>
      <div className="flex gap-4">
        <button
          type="button"
          onClick={() => handleChange(field, true)}
          className={`px-4 py-2 rounded transition-colors ${formData[field] === true
            ? 'bg-blue-600 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
        >
          {option_yes ? option_yes : 'Yes'}
        </button>
        <button
          type="button"
          onClick={() => handleChange(field, false)}
          className={`px-4 py-2 rounded transition-colors ${formData[field] === false
            ? 'bg-blue-600 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
        >
          {option_no ? option_no : 'No'}
        </button>
        {/* <button
          type="button"
          onClick={() => handleChange(field, null)}
          className={`px-4 py-2 rounded transition-colors ${
            formData[field] === null
              ? 'bg-blue-600 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          Unknown
        </button> */}
      </div>
    </div>
  );

  BooleanField.propTypes = {
    label: PropTypes.string.isRequired,
    field: PropTypes.string.isRequired,
    tooltip: PropTypes.string,
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow mb-6">
      <h2 className="card-title text-lg font-bold mb-4 flex items-center gap-2">
        PEP Risk Factors - Manual
        <span className="relative group text-gray-400 hover:text-gray-600 cursor-pointer text-sm">
          ⓘ
          <div className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-60 p-3 rounded-md
                          bg-blue-50 text-blue-800 text-sm shadow-lg border border-blue-100
                          opacity-0 group-hover:opacity-100 pointer-events-none">
            Note: Additional risk factors will be automatically
            extracted from the procedure transcript.
          </div>
        </span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Demographics */}
        <div>
          <h3 className="text-lg font-semibold mb-3 text-gray-700">Demographics</h3>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Age (years)
            </label>
            <input
              type="number"
              min="0"
              max="120"
              value={formData.age_years}
              onChange={(e) => handleChange('age_years', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter age"
            />
          </div>

          <BooleanField option_yes={"Male"} option_no={"Female"}
            label="Gender"
            field="gender_male"
          />

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              BMI
            </label>
            <input
              type="number"
              step="0.1"
              min="10"
              max="80"
              value={formData.bmi}
              onChange={(e) => handleChange('bmi', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Enter BMI"
            />
          </div>
          {/* Procedure Factors */}
          <div className="md:col-span-2">
            <h3 className="text-lg font-semibold mb-3 text-gray-700">Procedure Factors</h3>
            <BooleanField
              label="Trainee Involvement"
              field="trainee_involvement"
            />
          </div>
        </div>

        {/* Medical History */}
        <div>
          <h3 className="text-lg font-semibold mb-3 text-gray-700">Medical History</h3>

          <BooleanField
            label="History of PEP"
            field="history_of_pep"
          />

          <BooleanField
            label="History of Recurrent Pancreatitis"
            field="hx_of_recurrent_pancreatitis"
          />

          <BooleanField
            label="Sphincter of Oddi Dysfunction (SOD)"
            field="sod"
          />

          <BooleanField
            label="Prior Cholecystectomy"
            field="cholecystectomy"
          />

          <BooleanField
            label="Pancreato-Biliary Malignancy"
            field="pancreo_biliary_malignancy"
          />
        </div>
      </div>
    </div>
  );
};

PEPRiskManualInput.propTypes = {
  onDataChange: PropTypes.func.isRequired,
  initialData: PropTypes.object,
};
