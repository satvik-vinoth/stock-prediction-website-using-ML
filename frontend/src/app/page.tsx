"use client";

import { useState } from "react";
import Header from "@/components/header";
import FrontPage from "@/components/frontpage";
import dynamic from "next/dynamic";
const CompanySelector = dynamic(() => import("@/components/CompanySelector"), { ssr: false });
import ModelSelector from "@/components/ModelSelector";

export default function Home() {
  const [selectedCompany, setSelectedCompany] = useState("AAPL");
  return (
    
    <div className="bg-[#1c3b35] min-h-screen">
      <section id="home">
      </section>
      <Header/>
      <FrontPage/>
      <section id="company">
      </section>
      <CompanySelector onCompanySelected={setSelectedCompany} />
      <section id="prediction">
      </section>
      <ModelSelector company={selectedCompany} />
      {/*<section className="w-full max-w-4xl mt-12 px-4" id="prediction">
        <PredictionDisplay />
      </section>*/}

      <footer className="w-full text-center text-sm text-gray-500 py-8" id="contact">
        &copy; {new Date().getFullYear()} Stock Vision -1234567890. All rights reserved.
      </footer>
    </div>
  );
}
