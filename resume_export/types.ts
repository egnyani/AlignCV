export interface Contact {
  name: string;
  phone: string;
  email: string;
  linkedin: string;
  github?: string;
}

export interface Experience {
  id: string;
  company: string;
  title: string;
  location: string;
  start: string;
  end: string;
  bullets: string[];
}

export interface Education {
  degree: string;
  school: string;
  location: string;
  start: string;
  end: string;
}

export interface Project {
  id: string;
  name: string;
  date: string;
  bullets: string[];
  tags: string[];
}

export interface ResumeData {
  contact: Contact;
  summary: string[];
  experience: Experience[];
  education: Education[];
  skills: Record<string, string[]>;
  projects: Project[];
}
