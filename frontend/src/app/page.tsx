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
    
    <div className="bg-[#1c3b35] min-h-screen relative overflow-hidden">
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

      <footer className="w-full text-center text-sm text-gray-500 py-8" id="contact">
        &copy; {new Date().getFullYear()} Stock Vision -1234567890. All rights reserved.
      </footer>
    </div>
  );
}
