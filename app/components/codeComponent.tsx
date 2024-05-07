// components/PythonCode.tsx
"use client";
import React from "react";
import { Highlight, themes } from "prism-react-renderer"; // You can choose any other available theme

interface CodeComponentProps {
  code: string;
  lang: string;
}

const CodeComponent: React.FC<CodeComponentProps> = ({ code, lang }) => {
  return (
    <Highlight theme={themes.dracula} code={code} language={lang}>
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <pre className={className} style={style}>
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })}>
              {line.map((token, key) => (
                <span key={key} {...getTokenProps({ token })} />
              ))}
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
};

export default CodeComponent;
