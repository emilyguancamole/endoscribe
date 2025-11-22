import argparse
import os
import pandas as pd
from drafters import ColonoscopyDrafter, EUSDrafter, ERCPDrafter, EGDDrafter


def main():
    """
    Will overwrite any existing reports with same name.

    Example use:
        python drafter.py --procedure=col --pred_csv=longform/long-10-2025_colonoscopies.csv --polyp_csv=results/col/longform/long-10-2025_polyps.csv --output_dir=drafters/results/col/longform --samples_to_process all 

        python drafter.py --procedure=eus --pred_csv=longform/long-10-2025.csv --output_dir=drafters/results/eus/longform --samples_to_process mass02 cancer07

        python drafter.py --procedure=ercp --pred_csv=longform/long-10-2025.csv --output_dir=drafters/results/ercp/longform --samples_to_process bdstricture01 bdstone01

        python drafter.py --procedure=egd --pred_csv=longform/long-10-2025.csv --output_dir=drafters/results/egd/longform --samples_to_process egd01 egd02 egd03
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--procedure', type=str, required=True, choices=['col', 'eus', 'ercp', 'egd'])
    parser.add_argument('--pred_csv', required=True, help="Predictions csv path WITHIN results/{procedure}/extractions/")
    parser.add_argument('--polyp_csv', required=False, help="Path to corresponding polyp predictions csv, for colonoscopy only")
    parser.add_argument('--patients_data', default="data/patients.csv", help="Path to patient info data from RedCap")
    parser.add_argument('--procedures_data', default="data/procedures.csv", help="Path to patient info data from RedCap")
    parser.add_argument('--output_dir', required=True, help="Directory to save reports")
    parser.add_argument('--procedure_subtype', required=False, help="Optional: force a procedure subtype (eg 'ercp_cholangioscopy') for all samples")
    parser.add_argument('--transcripts', default='transcription/eus_transcripts/initial_whisper_lg_v3.csv', help="File with AI transcripts so we can add to report for reference")
    parser.add_argument('--samples_to_process', nargs='*', help="List of sample numbers to process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load transcripts if provided so we can include them in the saved package
    transcripts_df = None
    if args.transcripts:
        try:
            transcripts_df = pd.read_csv(args.transcripts, index_col=0, dtype={0: str})
        except Exception:
            transcripts_df = None

    # Load dataframes
    pred_csv_path = os.path.join("results", args.procedure, "extractions", args.pred_csv) # all pred csv paths results/procedure_name/extractions/
    pred_df = pd.read_csv(pred_csv_path, index_col=0, dtype={0: str})
    patients_df = pd.read_csv(args.patients_data, index_col=0, dtype={0: str})
    procedures_df = pd.read_csv(args.procedures_data, index_col=0, dtype={0: str})

    polyp_df = None
    if args.procedure == "col":
        if not args.polyp_csv:
            raise ValueError("For colonoscopy reports, provide a path to the polyp predictions, --polyp_csv")
        polyp_df = pd.read_csv(args.polyp_csv, index_col=0, dtype={0: str})

    if not args.samples_to_process:
        raise ValueError("Provide --samples_to_process, or use 'all' to process every sample in the predictions CSV")

    if args.samples_to_process[0] == "all":
        samples_to_process = pred_df.index.tolist()
    else:
        samples_to_process = args.samples_to_process

    # Map procedure shortname to drafter class
    drafter_classes = {
        "col": ColonoscopyDrafter,
        "eus": EUSDrafter,
        "ercp": ERCPDrafter,
        "egd": EGDDrafter,
    }

    Drafter = drafter_classes[args.procedure]

    for sample in samples_to_process:
        if args.procedure == "col":
            drafter = Drafter(sample, pred_df, patients_df, procedures_df, polyp_df)
        else:
            drafter = Drafter(sample, pred_df, patients_df, procedures_df)

        # Attach transcripts dataframe to drafter for packaging (may be None)
        drafter.transcripts_df = transcripts_df
        #? If CLI override provided, use it; otherwise try procedures_df column
        if args.procedure_subtype:
            drafter.procedure_variant = args.procedure_subtype
        else:
            try:
                subtype = procedures_df.loc[sample].get('procedure_subtype')
                if subtype and str(subtype).strip() and str(subtype).strip().lower() != 'nan':
                    drafter.procedure_variant = subtype
            except Exception:
                # leave default
                pass

        doc = drafter.draft_doc()
        out_fp = os.path.join(args.output_dir, f"{sample}.docx")
        doc.save(out_fp)
        print(f"Report for '{sample}' created at {out_fp}")
        # 11/16 Also save a JSON package with the transcript, extracted fields, rendered sections, and provenance
        try:
            from pathlib import Path
            import json
            package_fp = os.path.join(args.output_dir, f"{sample}_package.json")
            def _safe_serialize(obj):
                try:
                    return json.loads(json.dumps(obj))
                except Exception:
                    return str(obj)

            raw_fields = None
            try:
                raw_fields = drafter.sample_df.to_dict()
            except Exception:
                raw_fields = dict(drafter.sample_df)

            normalized = None
            validation_error = None
            # Try to validate/normalize against a generated pydantic model if available
            try:
                proc_variant = getattr(drafter, 'procedure_variant', None) or getattr(drafter, 'PROCEDURE_VARIANT', None)
                if proc_variant:
                    # module path like data_models.generated_ercp_base_model
                    mod_name = f"data_models.generated_{proc_variant}_model"
                    model_mod = __import__(mod_name, fromlist=['*'])
                    # class name like ErcpBaseData
                    class_name = ''.join([p.title() for p in proc_variant.split('_')]) + 'Data'
                    ModelClass = getattr(model_mod, class_name, None)
                    if ModelClass:
                        try:
                            obj = ModelClass(**raw_fields)
                            normalized = obj.model_dump()
                        except Exception as e:
                            validation_error = str(e)
            except Exception:
                pass

            # Transcript
            transcript_text = None
            try:
                if hasattr(drafter, 'transcripts_df') and drafter.transcripts_df is not None and drafter.sample in drafter.transcripts_df.index:
                    print("linking transcript for sample", drafter.sample)
                    transcript_text = str(drafter.transcripts_df.loc[drafter.sample].get('pred_transcript', '') or '')
                    print("transcript_text:", transcript_text)

            except Exception:
                transcript_text = None

            package = {
                'sample_id': sample,
                'meta': {
                    'procedure': args.procedure,
                    'procedure_variant': getattr(drafter, 'procedure_variant', None) or getattr(drafter, 'PROCEDURE_VARIANT', None),
                },
                'original_transcript': transcript_text,
                'extracted_fields': {
                    'raw': _safe_serialize(raw_fields),
                    'normalized': _safe_serialize(normalized) if normalized is not None else None,
                    'validation_error': validation_error,
                },
                'rendered_sections': _safe_serialize(getattr(drafter, 'rendered_sections', None)),
                'rendered_note': getattr(drafter, 'rendered_note', None),
                'docx_path': str(Path(out_fp).resolve()),
                'provenance': getattr(drafter, 'provenance', None)
            }

            with open(package_fp, 'w') as pf:
                json.dump(package, pf, indent=2, ensure_ascii=False)
            print(f"Saved package JSON for '{sample}' at {package_fp}")
        except Exception as e:
            print(f"[WARN] Failed to save package JSON for {sample}: {e}")


if __name__ == "__main__":
    main()