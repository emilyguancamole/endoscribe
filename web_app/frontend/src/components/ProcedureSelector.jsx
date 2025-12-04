import { PROCEDURE_TYPES } from '../constants';
import { ListFilter } from 'lucide-react';

export function ProcedureSelector({ value, onChange }) {
  return (
    <div className="card bg-base-100 shadow-sm mb-4 border border-base-200">
      <div className="card-body flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 py-3">
        <div className="flex items-center gap-2">
          <ListFilter className="w-5 h-5 text-primary" />
          <h2 className="card-title text-lg font-semibold">Procedure Type</h2>
        </div>

        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="select select-bordered w-full sm:w-48"
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
