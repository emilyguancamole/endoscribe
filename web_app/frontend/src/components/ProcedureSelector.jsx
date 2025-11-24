import { PROCEDURE_TYPES } from '../constants';

export function ProcedureSelector({ value, onChange }) {
  return (
    <div className="card bg-base-100 shadow-xl mb-4">
      <div className="card-body">
        <h2 className="card-title">Select Procedure Type</h2>
        <select 
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="select select-bordered w-full max-w-xs"
        >
          {PROCEDURE_TYPES.map(type => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
