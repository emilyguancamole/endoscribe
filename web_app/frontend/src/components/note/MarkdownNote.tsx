import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Textarea } from "@/components/ui/textarea";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";

type MarkdownNoteProps = {
    noteContent: string;
    setNoteContent: (s: string) => void;
    defaultEditing?: boolean;
};

const sanitizeSchema = {
    ...defaultSchema,
    tagNames: [
        ...(defaultSchema.tagNames || []),
        "h1", "h2", "h3", "h4", "h5", "h6",
        "table", "thead", "tbody", "tr", "th", "td",
    ],
};

export default function MarkdownNote({ noteContent, setNoteContent, defaultEditing = false }: MarkdownNoteProps) {
    const [isEditing, setIsEditing] = useState<boolean>(defaultEditing);

    return (
        <div>
            <div className="ml-6 mt-6 items-center justify-between">
                <button
                    className="px-3 py-1 rounded-md border border-border/50 text-large font-medium bg-muted hover:bg-white"
                    onClick={() => setIsEditing((v) => !v)}
                >
                    {isEditing ? "Preview" : "Edit"}
                </button>
            </div>

            {isEditing ? (
                <Textarea
                    className="min-h-[500px] resize-y border-0 focus-visible:ring-0 p-6 text-base leading-relaxed font-normal"
                    value={noteContent}
                    onChange={(e) => setNoteContent(e.target.value)}
                />
            ) : (
                <div className="prose prose-slate max-w-none p-6 min-h-[500px] overflow-auto bg-white">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[
                            [rehypeSanitize, sanitizeSchema],
                            rehypeHighlight,
                        ]}
                    >
                        {noteContent || "No note content"}
                    </ReactMarkdown>
                </div>
            )}
        </div>
    );
}
