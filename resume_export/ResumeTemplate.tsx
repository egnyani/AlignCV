import type { ResumeData } from "./types";
import {
  pageStyle,
  nameStyle,
  contactRowStyle,
  sectionHeaderStyle,
  summaryListStyle,
  summaryItemStyle,
  roleBlockStyle,
  companyDateRowStyle,
  companyNameStyle,
  dateStyle,
  roleTitleStyle,
  bulletListStyle,
  bulletItemStyle,
  skillRowStyle,
  skillLabelStyle,
  skillValuesStyle,
  eduBlockStyle,
  eduSchoolRowStyle,
  eduSchoolStyle,
  eduDegreeRowStyle,
  eduDegreeStyle,
  eduLocationStyle,
  projectBlockStyle,
  projectHeaderRowStyle,
  projectNameStyle,
  linkStyle,
  renderInlineBold,
} from "./resumeLayout";

interface ResumeTemplateProps {
  data: ResumeData;
}

function ensureHref(value: string): string {
  if (/^https?:\/\//i.test(value)) return value;
  return `https://${value}`;
}

export function ResumeTemplate({ data }: ResumeTemplateProps) {
  return (
    <div style={pageStyle}>
      <h1 style={nameStyle}>{data.contact.name}</h1>

      <p style={contactRowStyle}>
        {data.contact.phone}
        {"  |  "}
        {data.contact.email}
        {"  |  "}
        <a href={ensureHref(data.contact.linkedin)} style={linkStyle}>
          LinkedIn
        </a>
        {data.contact.github && (
          <>
            {"  |  "}
            <a href={ensureHref(data.contact.github)} style={linkStyle}>
              Github
            </a>
          </>
        )}
      </p>

      <section>
        <h2 style={sectionHeaderStyle}>Summary</h2>
        <ul style={summaryListStyle}>
          {data.summary.map((point, i) => (
            <li key={i} style={summaryItemStyle}>
              {renderInlineBold(point)}
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 style={sectionHeaderStyle}>Experience</h2>
        {data.experience.map((role) => (
          <div key={role.id} style={roleBlockStyle}>
            <div style={companyDateRowStyle}>
              <span style={companyNameStyle}>{role.company}</span>
              <span style={dateStyle}>
                {role.start} - {role.end}
              </span>
            </div>
            <div style={roleTitleStyle}>{role.title}</div>
            {role.location ? (
              <div style={{ ...roleTitleStyle, fontStyle: "normal", fontSize: "10pt" }}>
                {role.location}
              </div>
            ) : null}
            <ul style={bulletListStyle}>
              {role.bullets.map((bullet, index) => (
                <li key={index} style={bulletItemStyle}>
                  {renderInlineBold(bullet)}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </section>

      <section>
        <h2 style={sectionHeaderStyle}>Technical Skills</h2>
        {Object.entries(data.skills).map(([category, values]) => (
          <div key={category} style={skillRowStyle}>
            <span style={skillLabelStyle}>{category}:</span>
            <span style={skillValuesStyle}> {values.join(", ")}</span>
          </div>
        ))}
      </section>

      <section>
        <h2 style={sectionHeaderStyle}>Education</h2>
        {data.education.map((entry) => (
          <div key={`${entry.school}-${entry.degree}`} style={eduBlockStyle}>
            <div style={eduSchoolRowStyle}>
              <span style={eduSchoolStyle}>{entry.school}</span>
              <span style={dateStyle}>
                {entry.start} - {entry.end}
              </span>
            </div>
            <div style={eduDegreeRowStyle}>
              <span style={eduDegreeStyle}>{entry.degree}</span>
              <span style={eduLocationStyle}>{entry.location}</span>
            </div>
          </div>
        ))}
      </section>

      {data.projects && data.projects.length > 0 && (
        <section>
          <h2 style={sectionHeaderStyle}>Projects</h2>
          {data.projects.map((project) => (
            <div key={project.id} style={projectBlockStyle}>
              <div style={projectHeaderRowStyle}>
                <span style={projectNameStyle}>{project.name}</span>
                <span style={dateStyle}>{project.date}</span>
              </div>
              <ul style={bulletListStyle}>
                {project.bullets.map((bullet, index) => (
                  <li key={index} style={bulletItemStyle}>
                    {renderInlineBold(bullet)}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
