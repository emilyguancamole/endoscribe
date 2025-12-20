export interface ProcedureType {
  id: string;
  name: string;
  subtypes: string[];
}

export const procedureTypes: ProcedureType[] = [
  {
    id: "gi",
    name: "Gastroenterology",
    subtypes: ["col", "egd", "ercp", "eus"],
  },
];

export const procedureLabels: Record<string, string> = {
  ercp: "ERCP",
  col: "Colonoscopy - not yet implemented",
  egd: "EGD - not yet implemented",
  eus: "EUS - not yet implemented",
};

export const ERROR_DISPLAY_DURATION_MS = 5000;
