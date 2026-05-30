import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

interface SectionHeaderProps {
  icon: LucideIcon;
  iconColor?: string;
  iconBg?: string;
  title: string;
  titleColor?: string;
  subtitle: string;
  action?: ReactNode;
}

export function SectionHeader({
  icon: Icon,
  iconColor = "text-purple-400",
  iconBg = "bg-purple-500/10",
  title,
  titleColor = "text-white",
  subtitle,
  action,
}: SectionHeaderProps) {
  return (
    <div className="flex items-center justify-between border-b border-white/5 pb-6">
      <div className="space-y-1">
        <h2
          className={`text-3xl font-bold flex items-center gap-3 ${titleColor}`}
        >
          <span className={`p-2 rounded-lg ${iconBg} ${iconColor}`}>
            <Icon className="w-6 h-6" />
          </span>
          {title}
        </h2>
        <p className="text-gray-500">{subtitle}</p>
      </div>
      {action}
    </div>
  );
}
