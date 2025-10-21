import React from 'react'
export const MenuIcon = ({size=20}) => (
<svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
</svg>
)


export const SendIcon = ({size=18}) => (
<svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M3 11l18-8-8 18-3-7-7-3z" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
</svg>
)


export const LogoIcon = ({size=28}) => (
<svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="g" x1="0" x2="1">
<stop offset="0" stopColor="#00e0c6" />
<stop offset="1" stopColor="#ad53ff" />
</linearGradient>
</defs>
<rect x="2" y="2" width="20" height="20" rx="6" fill="url(#g)" opacity="0.15"/>
<path d="M7 12c1.2-2 3-4 5-4s3.8 2 5 4" stroke="url(#g)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
<circle cx="12" cy="12" r="2" fill="url(#g)" />
</svg>
)