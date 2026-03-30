import type { CSSProperties, ReactNode } from "react";

export const PAGE_PADDING = 32;

export const PDF_CONFIG = {
  width: "8.5in",
  margin: { top: "0px", bottom: "0px", left: "0px", right: "0px" },
  printBackground: true,
} as const;

export const HTML_RESET_CSS =
  "* { margin: 0; padding: 0; box-sizing: border-box; } body { background: white; }";

export function renderInlineBold(text: string): ReactNode {
  const parts = text.split(/\*\*(.+?)\*\*/g);
  if (parts.length === 1) return text;
  return (
    <>
      {parts.map((part, i) => (i % 2 === 1 ? <strong key={i}>{part}</strong> : part))}
    </>
  );
}

export const pageStyle: CSSProperties = {
  width: "100%",
  maxWidth: "750px",
  margin: "0 auto",
  backgroundColor: "#ffffff",
  padding: `${PAGE_PADDING}px`,
  boxSizing: "border-box",
  fontFamily: "'Times New Roman', Times, Georgia, serif",
  fontSize: "11pt",
  lineHeight: 1.35,
  color: "#000000",
};

export const nameStyle: CSSProperties = {
  fontSize: "18pt",
  fontWeight: 700,
  textAlign: "center",
  color: "#000000",
  margin: "0 0 3px 0",
  letterSpacing: "0.5px",
};

export const contactRowStyle: CSSProperties = {
  fontSize: "10pt",
  color: "#457885",
  textAlign: "center",
  margin: "0 0 6px 0",
  letterSpacing: "0.1px",
};

export const sectionHeaderStyle: CSSProperties = {
  fontSize: "10.5pt",
  fontWeight: 700,
  fontVariant: "small-caps",
  letterSpacing: "0.5px",
  color: "#000000",
  borderBottom: "1px solid #000000",
  paddingBottom: "1px",
  margin: "7px 0 3px 0",
  textTransform: "uppercase",
};

export const summaryListStyle: CSSProperties = {
  listStyleType: "disc",
  marginLeft: "20px",
  marginTop: "0",
  padding: "0",
};

export const summaryItemStyle: CSSProperties = {
  fontSize: "11pt",
  lineHeight: "1.45",
  color: "#000000",
  marginBottom: "2px",
  textAlign: "justify",
};

export const roleBlockStyle: CSSProperties = {
  marginBottom: "6px",
};

export const companyDateRowStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  marginBottom: "0px",
};

export const companyNameStyle: CSSProperties = {
  fontSize: "10.5pt",
  fontWeight: 700,
  color: "#000000",
};

export const dateStyle: CSSProperties = {
  fontSize: "11pt",
  color: "#000000",
  whiteSpace: "nowrap",
  fontWeight: 400,
};

export const roleTitleStyle: CSSProperties = {
  fontSize: "11pt",
  fontStyle: "italic",
  fontWeight: 400,
  color: "#000000",
  marginBottom: "1px",
};

export const bulletListStyle: CSSProperties = {
  listStyleType: "disc",
  marginLeft: "20px",
  padding: "0",
  marginTop: "1px",
};

export const bulletItemStyle: CSSProperties = {
  fontSize: "11pt",
  lineHeight: "1.45",
  marginBottom: "2px",
  textAlign: "justify",
};

export const skillRowStyle: CSSProperties = {
  fontSize: "11pt",
  lineHeight: "1.55",
  marginBottom: "1px",
};

export const skillLabelStyle: CSSProperties = {
  fontWeight: 700,
  color: "#000000",
};

export const skillValuesStyle: CSSProperties = {
  fontWeight: 400,
  color: "#000000",
};

export const eduBlockStyle: CSSProperties = {
  marginBottom: "4px",
};

export const eduSchoolRowStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
};

export const eduSchoolStyle: CSSProperties = {
  fontSize: "10.5pt",
  fontWeight: 700,
  color: "#000000",
};

export const eduDegreeRowStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
};

export const eduDegreeStyle: CSSProperties = {
  fontSize: "11pt",
  color: "#000000",
};

export const eduLocationStyle: CSSProperties = {
  fontSize: "11pt",
  color: "#555555",
  whiteSpace: "nowrap",
};

export const projectBlockStyle: CSSProperties = {
  marginBottom: "6px",
};

export const projectHeaderRowStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
};

export const projectNameStyle: CSSProperties = {
  fontSize: "10.5pt",
  fontWeight: 700,
  color: "#000000",
};

export const linkStyle: CSSProperties = {
  color: "#457885",
  textDecoration: "none",
};
